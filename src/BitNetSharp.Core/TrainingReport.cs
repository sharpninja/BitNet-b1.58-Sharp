namespace BitNetSharp.Core;

public sealed record TrainingReport(
    IReadOnlyList<double> LossHistory,
    int SamplesSeen,
    int Epochs,
    int NegativeWeights,
    int ZeroWeights,
    int PositiveWeights)
{
    public double AverageLoss => LossHistory.Count == 0 ? 0d : LossHistory.Average();
}
