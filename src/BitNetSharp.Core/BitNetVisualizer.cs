using System.Text;

namespace BitNetSharp.Core;

public sealed class BitNetVisualizer
{
    public VisualizationReport CreateReport(TrainingReport trainingReport)
    {
        ArgumentNullException.ThrowIfNull(trainingReport);

        var lossChart = RenderLossChart(trainingReport.LossHistory);
        var histogram = RenderWeightHistogram(trainingReport);
        var csv = RenderCsv(trainingReport);

        return new VisualizationReport(lossChart, histogram, csv);
    }

    private static string RenderLossChart(IReadOnlyList<double> lossHistory)
    {
        var lines = new List<string> { "Loss by epoch" };
        for (var index = 0; index < lossHistory.Count; index++)
        {
            var value = lossHistory[index];
            var barLength = Math.Max(1, (int)Math.Round((1d - value) * 20d));
            lines.Add($"Epoch {index + 1,2}: {new string('#', barLength)} {value:0.000}");
        }

        return string.Join(Environment.NewLine, lines);
    }

    private static string RenderWeightHistogram(TrainingReport trainingReport)
    {
        var max = new[] { trainingReport.NegativeWeights, trainingReport.ZeroWeights, trainingReport.PositiveWeights }.Max();
        max = Math.Max(max, 1);

        return string.Join(
            Environment.NewLine,
            new[]
            {
                "Ternary weight distribution",
                FormatBar("-1", trainingReport.NegativeWeights, max),
                FormatBar(" 0", trainingReport.ZeroWeights, max),
                FormatBar("+1", trainingReport.PositiveWeights, max)
            });
    }

    private static string RenderCsv(TrainingReport trainingReport)
    {
        var builder = new StringBuilder();
        builder.AppendLine("epoch,loss");

        for (var index = 0; index < trainingReport.LossHistory.Count; index++)
        {
            builder.AppendLine($"{index + 1},{trainingReport.LossHistory[index]:0.000}");
        }

        return builder.ToString();
    }

    private static string FormatBar(string label, int value, int max)
    {
        var width = Math.Max(1, (int)Math.Round(value / (double)max * 20d));
        return $"{label}: {new string('#', width)} {value}";
    }
}
