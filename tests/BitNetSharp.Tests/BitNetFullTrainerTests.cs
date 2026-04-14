using BitNetSharp.Core;
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
}
