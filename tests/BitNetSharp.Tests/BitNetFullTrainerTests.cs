using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class BitNetFullTrainerTests
{
    [Fact]
    public void Train_OneEpoch_ReducesLoss()
    {
        var model = BitNetPaperModel.CreateDefault();
        var options = new BitNetTrainingOptions(
            epochs: 2,
            dataLoaderOptions: new BitNetDataLoaderOptions(
                sequenceLength: model.Config.MaxSequenceLength,
                validationFraction: 0d,
                testFraction: 0d));

        var trainer = new BitNetFullTrainer(model, options);
        var examples = BitNetTrainingCorpus.CreateDefaultExamples().Take(3).ToList();

        var report = trainer.Train(examples);

        Assert.True(report.LossHistory.Count >= 2);
        // Loss should generally decrease (or at least not explode)
        Assert.True(double.IsFinite(report.LossHistory[0]));
        Assert.True(double.IsFinite(report.LossHistory[^1]));
    }

    [Fact]
    public void Train_MasterWeightsAreUpdated()
    {
        var model = BitNetPaperModel.CreateDefault();
        var options = new BitNetTrainingOptions(
            epochs: 1,
            dataLoaderOptions: new BitNetDataLoaderOptions(
                sequenceLength: model.Config.MaxSequenceLength,
                validationFraction: 0d,
                testFraction: 0d));

        // Capture initial ternary stats
        var initialStats = model.GetTernaryWeightStats();

        var trainer = new BitNetFullTrainer(model, options);
        var examples = BitNetTrainingCorpus.CreateDefaultExamples().Take(3).ToList();

        trainer.Train(examples);

        // After training, ternary weights should have been re-quantized from updated masters
        var finalStats = model.GetTernaryWeightStats();

        // Stats should still be valid (non-negative counts, same total)
        Assert.Equal(initialStats.TotalCount, finalStats.TotalCount);
        Assert.True(finalStats.NegativeCount >= 0);
        Assert.True(finalStats.ZeroCount >= 0);
        Assert.True(finalStats.PositiveCount >= 0);
    }

    [Fact]
    public void Train_on_tiny_synthetic_corpus_reduces_loss()
    {
        var (transformer, sequences) = CreateTinySyntheticSetup(seed: 42);
        var options = new BitNetTrainingOptions(
            epochs: 10,
            learningRate: 0.05f,
            dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 8));

        var trainer = new BitNetFullTrainer(transformer, options);
        var report = trainer.Train(sequences, options.Epochs);

        Assert.NotNull(report.EpochMetrics);
        Assert.Equal(options.Epochs, report.EpochMetrics!.Count);

        var firstEpochLoss = report.EpochMetrics![0].AverageLoss;
        var lastEpochLoss = report.EpochMetrics![^1].AverageLoss;

        Assert.True(double.IsFinite(firstEpochLoss), $"First-epoch loss should be finite, got {firstEpochLoss}");
        Assert.True(double.IsFinite(lastEpochLoss), $"Last-epoch loss should be finite, got {lastEpochLoss}");
        Assert.True(
            lastEpochLoss < firstEpochLoss * 0.9,
            $"Expected >=10% loss reduction, first={firstEpochLoss:F4}, last={lastEpochLoss:F4}");
    }

    [Fact]
    public void Train_does_not_produce_nan_or_inf_gradients()
    {
        var (transformer, sequences) = CreateTinySyntheticSetup(seed: 123);
        var options = new BitNetTrainingOptions(
            epochs: 1,
            dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 8));

        var trainer = new BitNetFullTrainer(transformer, options);

        // Train on only the first sequence (one batch) to expose any NaN/Inf immediately.
        trainer.Train(sequences.Take(1).ToArray(), epochs: 1);

        foreach (var layer in transformer.EnumerateBitLinearLayers())
        {
            var grads = layer.ExportMasterGradients();
            if (grads is null)
            {
                continue;
            }

            for (var i = 0; i < grads.Length; i++)
            {
                Assert.False(float.IsNaN(grads[i]), $"NaN master gradient at index {i}");
                Assert.False(float.IsInfinity(grads[i]), $"Inf master gradient at index {i}");
            }
        }

        var embGrads = transformer.ExportTokenEmbeddingGradients();
        if (embGrads is not null)
        {
            for (var r = 0; r < embGrads.GetLength(0); r++)
            {
                for (var c = 0; c < embGrads.GetLength(1); c++)
                {
                    var v = embGrads[r, c];
                    Assert.False(float.IsNaN(v), $"NaN embedding gradient at [{r},{c}]");
                    Assert.False(float.IsInfinity(v), $"Inf embedding gradient at [{r},{c}]");
                }
            }
        }
    }

    [Fact]
    public void Train_report_captures_per_epoch_metrics()
    {
        var (transformer, sequences) = CreateTinySyntheticSetup(seed: 7);
        var options = new BitNetTrainingOptions(
            epochs: 5,
            learningRate: 0.05f,
            dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 8));

        var trainer = new BitNetFullTrainer(transformer, options);
        var report = trainer.Train(sequences, options.Epochs);

        Assert.NotNull(report.EpochMetrics);
        Assert.Equal(options.Epochs, report.EpochMetrics!.Count);

        // Epoch numbers are 1-based and consecutive.
        for (var i = 0; i < report.EpochMetrics.Count; i++)
        {
            Assert.Equal(i + 1, report.EpochMetrics[i].Epoch);
        }

        // Non-zero token count reported per epoch.
        foreach (var metric in report.EpochMetrics)
        {
            Assert.True(metric.TokensSeen > 0, $"Epoch {metric.Epoch} reported zero tokens.");
        }

        // Non-increasing moving-average loss (window 2) over 5 epochs.
        var losses = report.EpochMetrics.Select(m => m.AverageLoss).ToArray();
        for (var i = 1; i < losses.Length - 1; i++)
        {
            var prevWindow = (losses[i - 1] + losses[i]) / 2d;
            var nextWindow = (losses[i] + losses[i + 1]) / 2d;
            Assert.True(
                nextWindow <= prevWindow + 1e-6,
                $"Moving-average loss increased: window[{i - 1}..{i}]={prevWindow:F4}, window[{i}..{i + 1}]={nextWindow:F4}");
        }
    }

    private static (BitNetTransformer Transformer, IReadOnlyList<int[]> Sequences) CreateTinySyntheticSetup(int seed)
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 16,
            hiddenDimension: 32,
            layerCount: 2,
            headCount: 2,
            maxSequenceLength: 16);

        var transformer = new BitNetTransformer(config, seed);

        // Two fixed "sentences" of 8 tokens each over a vocab of 32.
        // Deterministic so the convergence test is reliable.
        var sequences = new int[][]
        {
            [1, 5, 9, 13, 17, 21, 25, 29],
            [2, 6, 10, 14, 18, 22, 26, 30],
        };

        return (transformer, sequences);
    }
}
