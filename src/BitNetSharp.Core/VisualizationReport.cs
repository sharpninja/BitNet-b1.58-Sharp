namespace BitNetSharp.Core;

public sealed record VisualizationReport(
    string LossChart,
    string WeightHistogram,
    string Csv);
