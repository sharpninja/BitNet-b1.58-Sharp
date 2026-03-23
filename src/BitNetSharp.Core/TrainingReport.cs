namespace BitNetSharp.Core;

public sealed record TrainingEpochMetrics(
    int Epoch,
    double AverageLoss,
    int SamplesSeen,
    int TokensSeen,
    double? ValidationPerplexity = null);

public sealed record TrainingEvaluationSummary(
    string Dataset,
    int Samples,
    double AverageCrossEntropy,
    double Perplexity);

public sealed record TrainingCheckpointSummary(
    int Epoch,
    int SamplesSeen,
    string Path);

public sealed record TrainingReport(
    IReadOnlyList<double> LossHistory,
    int SamplesSeen,
    int Epochs,
    int NegativeWeights,
    int ZeroWeights,
    int PositiveWeights,
    IReadOnlyList<TrainingEpochMetrics>? EpochMetrics = null,
    IReadOnlyList<TrainingEvaluationSummary>? EvaluationSummaries = null,
    IReadOnlyList<TrainingCheckpointSummary>? Checkpoints = null,
    string? TrainingDataset = null,
    string? ValidationDataset = null)
{
    public double AverageLoss => LossHistory.Count == 0 ? 0d : LossHistory.Average();

    public double? LatestValidationPerplexity => EpochMetrics?
        .LastOrDefault(static metric => metric.ValidationPerplexity is not null)?
        .ValidationPerplexity;
}
