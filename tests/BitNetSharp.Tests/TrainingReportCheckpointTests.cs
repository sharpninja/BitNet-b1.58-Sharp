using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class TrainingReportCheckpointTests
{
    [Fact]
    public void PaperModelTrainingReportIncludesPeriodicCheckpointSummaries()
    {
        var checkpointDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-training-checkpoints-{Guid.NewGuid():N}");
        Directory.CreateDirectory(checkpointDirectory);

        try
        {
            var examples = new[] { new TrainingExample("alpha beta", "gamma delta") };
            var model = BitNetPaperModel.CreateForTrainingCorpus(examples, VerbosityLevel.Quiet);

            var report = model.Train(
                examples,
                new BitNetTrainingOptions(
                    epochs: 3,
                    learningRate: 0.2f,
                    evaluationInterval: 0,
                    checkpointInterval: 1,
                    dataLoaderOptions: new BitNetDataLoaderOptions(
                        sequenceLength: 8,
                        batchSize: 1,
                        validationFraction: 0d,
                        testFraction: 0d,
                        dropLast: false),
                    trainingDatasetName: "unit-train",
                    checkpointDirectory: checkpointDirectory,
                    checkpointPrefix: "cadence-checkpoint"));

            Assert.NotNull(report.Checkpoints);
            Assert.Collection(
                report.Checkpoints!,
                checkpoint =>
                {
                    Assert.Equal(1, checkpoint.Epoch);
                    Assert.True(checkpoint.SamplesSeen > 0);
                    Assert.EndsWith("cadence-checkpoint-epoch1.bitnet.json", checkpoint.Path, StringComparison.Ordinal);
                    Assert.True(File.Exists(checkpoint.Path));
                },
                checkpoint =>
                {
                    Assert.Equal(2, checkpoint.Epoch);
                    Assert.True(checkpoint.SamplesSeen > 0);
                    Assert.EndsWith("cadence-checkpoint-epoch2.bitnet.json", checkpoint.Path, StringComparison.Ordinal);
                    Assert.True(File.Exists(checkpoint.Path));
                },
                checkpoint =>
                {
                    Assert.Equal(3, checkpoint.Epoch);
                    Assert.True(checkpoint.SamplesSeen > 0);
                    Assert.EndsWith("cadence-checkpoint-epoch3.bitnet.json", checkpoint.Path, StringComparison.Ordinal);
                    Assert.True(File.Exists(checkpoint.Path));
                });
        }
        finally
        {
            if (Directory.Exists(checkpointDirectory))
            {
                Directory.Delete(checkpointDirectory, recursive: true);
            }
        }
    }

    [Fact]
    public void PaperModelTrainingReportUsesExternalEvaluationForEpochMetricsWhenProvided()
    {
        var examples = new[] { new TrainingExample("alpha beta", "gamma delta") };
        var model = BitNetPaperModel.CreateForTrainingCorpus(examples, VerbosityLevel.Quiet);

        var report = model.Train(
            examples,
            new BitNetTrainingOptions(
                epochs: 2,
                learningRate: 0.2f,
                evaluationInterval: 1,
                dataLoaderOptions: new BitNetDataLoaderOptions(
                    sequenceLength: 8,
                    batchSize: 1,
                    validationFraction: 0d,
                    testFraction: 0d,
                    dropLast: false),
                validationDatasetName: "external-valid",
                externalEvaluation: epoch => new TrainingEvaluationSummary("external-valid", 1, epoch, epoch + 10d)));

        Assert.NotNull(report.EpochMetrics);
        Assert.Collection(
            report.EpochMetrics!,
            metric => Assert.Equal(11d, metric.ValidationPerplexity),
            metric => Assert.Equal(12d, metric.ValidationPerplexity));
        Assert.Contains(report.EvaluationSummaries!, summary => summary.Dataset == "external-valid" && summary.Perplexity == 12d);
    }

    [Fact]
    public void PlainTextTrainingReportIncludesCheckpointSection()
    {
        var report = new TrainingReport(
            LossHistory: [0.9d, 0.4d],
            SamplesSeen: 8,
            Epochs: 2,
            NegativeWeights: 10,
            ZeroWeights: 5,
            PositiveWeights: 11,
            EpochMetrics:
            [
                new TrainingEpochMetrics(1, 0.9d, 4, 16, 12.5d),
                new TrainingEpochMetrics(2, 0.4d, 8, 32, 8.5d)
            ],
            EvaluationSummaries:
            [
                new TrainingEvaluationSummary("HeldOut", 2, 1.1d, 3d)
            ],
            Checkpoints:
            [
                new TrainingCheckpointSummary(1, 4, @"C:\temp\checkpoint-epoch1.bitnet.json")
            ],
            TrainingDataset: "unit-train",
            ValidationDataset: "unit-valid");
        var options = new TrainingCommandOptions(
            Dataset: "unit-train",
            Epochs: 2,
            EvaluateEvery: 1,
            EvaluationDataset: "unit-valid",
            CheckpointEvery: 1,
            CheckpointDirectory: @"C:\temp",
            CheckpointPrefix: "checkpoint",
            ReportPath: null,
            ReportFormat: TrainingCommandReportFormat.PlainText,
            CompactEvaluation: true,
            SaveCheckpoint: true,
            DryRun: false,
            HelpRequested: false);

        var document = TrainingCommandResultFormatter.FormatReportDocument(options, report);

        Assert.Contains("Checkpoints:", document, StringComparison.Ordinal);
        Assert.Contains(@"C:\temp\checkpoint-epoch1.bitnet.json", document, StringComparison.Ordinal);
        Assert.Contains("Evaluation summaries:", document, StringComparison.Ordinal);
        Assert.Contains("Epoch metrics:", document, StringComparison.Ordinal);
    }
}
