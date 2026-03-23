using System.Globalization;
using System.Text;
using System.Text.Json;
using BitNetSharp.Core;

namespace BitNetSharp.App;

public static class TrainingCommandResultFormatter
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public static string FormatUsage() =>
        string.Join(
            Environment.NewLine,
            [
                "Training command options:",
                "  --dataset <name|path>             Dataset alias or path to train on.",
                "  --epochs <count>                  Number of training epochs.",
                "  --eval-every <count>              Run evaluation every N epochs.",
                "  --validation-dataset <name|path>  Dataset alias or path for evaluation.",
                "  --checkpoint-every <count>        Save checkpoints every N epochs.",
                "  --checkpoint-dir <path>           Directory for checkpoint files.",
                "  --checkpoint-prefix <name>        Checkpoint file prefix.",
                "  --report-path <path>              Write the training report to a file.",
                "  --report-format <text|markdown|json>  Report file format.",
                "  --compact-eval                    Prefer compact evaluation fixtures.",
                "  --full-eval                       Use the full evaluation fixtures.",
                "  --save-checkpoint                 Enable checkpoint writing.",
                "  --no-save-checkpoint              Disable checkpoint writing.",
                "  --dry-run                         Parse options and print the plan only.",
                "  --help, -h                        Show this help text."
            ]);

    public static string FormatPlan(TrainingCommandOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var builder = new StringBuilder();
        builder.AppendLine("Training plan");
        builder.AppendLine($"Dataset: {options.Dataset}");
        builder.AppendLine($"Epochs: {options.Epochs}");
        builder.AppendLine($"Evaluation: {FormatEvaluationSchedule(options)}");
        builder.AppendLine($"Checkpoints: {FormatCheckpointSchedule(options)}");
        builder.AppendLine($"Report: {FormatReportSchedule(options)}");
        builder.AppendLine($"Evaluation fixtures: {(options.CompactEvaluation ? "compact" : "full")}");
        builder.AppendLine($"Dry run: {(options.DryRun ? "yes" : "no")}");
        return builder.ToString().TrimEnd();
    }

    public static string FormatConsoleSummary(TrainingCommandOptions options, TrainingReport report)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(report);

        var builder = new StringBuilder();
        builder.AppendLine("Training complete");
        builder.AppendLine($"Dataset: {options.Dataset}");
        builder.AppendLine($"Epochs: {options.Epochs}");
        builder.AppendLine($"Samples seen: {report.SamplesSeen}");
        builder.AppendLine($"Average loss: {report.AverageLoss.ToString("0.####", CultureInfo.InvariantCulture)}");
        builder.AppendLine($"Loss history: {FormatLossHistory(report.LossHistory)}");
        builder.AppendLine($"Training dataset: {FormatNullableText(report.TrainingDataset, options.Dataset)}");
        builder.AppendLine($"Validation dataset: {FormatNullableText(report.ValidationDataset, options.EvaluationDataset)}");
        builder.AppendLine($"Latest validation perplexity: {FormatNullableNumber(report.LatestValidationPerplexity)}");
        builder.AppendLine($"Weights: -{report.NegativeWeights} 0{report.ZeroWeights} +{report.PositiveWeights}");
        builder.AppendLine($"Evaluation: {FormatEvaluationSchedule(options)}");
        builder.AppendLine($"Checkpoints: {FormatCheckpointSchedule(options)}");
        return builder.ToString().TrimEnd();
    }

    public static string FormatReportDocument(TrainingCommandOptions options, TrainingReport report)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(report);

        return options.ReportFormat switch
        {
            TrainingCommandReportFormat.Markdown => FormatMarkdownReport(options, report),
            TrainingCommandReportFormat.Json => FormatJsonReport(options, report),
            _ => FormatPlainTextReport(options, report)
        };
    }

    public static string FormatReportWrittenMessage(string reportPath, TrainingCommandReportFormat format) =>
        $"Training report written to {reportPath} ({FormatReportFormat(format)}).";

    private static string FormatPlainTextReport(TrainingCommandOptions options, TrainingReport report)
    {
        var builder = new StringBuilder();
        builder.AppendLine(FormatConsoleSummary(options, report));
        AppendEpochMetricsPlainText(builder, report);
        AppendEvaluationSummariesPlainText(builder, report);
        AppendCheckpointPlainText(builder, report);
        return builder.ToString().TrimEnd();
    }

    private static string FormatMarkdownReport(TrainingCommandOptions options, TrainingReport report)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# Training report");
        builder.AppendLine();
        builder.AppendLine("## Plan");
        builder.AppendLine();
        builder.AppendLine($"- Dataset: `{options.Dataset}`");
        builder.AppendLine($"- Epochs: `{options.Epochs}`");
        builder.AppendLine($"- Evaluation: `{FormatEvaluationSchedule(options)}`");
        builder.AppendLine($"- Checkpoints: `{FormatCheckpointSchedule(options)}`");
        builder.AppendLine($"- Fixtures: `{(options.CompactEvaluation ? "compact" : "full")}`");
        builder.AppendLine($"- Training dataset: `{FormatNullableText(report.TrainingDataset, options.Dataset)}`");
        builder.AppendLine($"- Validation dataset: `{FormatNullableText(report.ValidationDataset, options.EvaluationDataset)}`");
        builder.AppendLine();
        builder.AppendLine("## Metrics");
        builder.AppendLine();
        builder.AppendLine("| Metric | Value |");
        builder.AppendLine("| --- | ---: |");
        builder.AppendLine($"| Samples seen | {report.SamplesSeen} |");
        builder.AppendLine($"| Epochs | {report.Epochs} |");
        builder.AppendLine($"| Average loss | {report.AverageLoss.ToString("0.####", CultureInfo.InvariantCulture)} |");
        builder.AppendLine($"| Latest validation perplexity | {FormatNullableNumber(report.LatestValidationPerplexity)} |");
        builder.AppendLine($"| Negative weights | {report.NegativeWeights} |");
        builder.AppendLine($"| Zero weights | {report.ZeroWeights} |");
        builder.AppendLine($"| Positive weights | {report.PositiveWeights} |");
        builder.AppendLine();
        AppendEpochMetricsMarkdown(builder, report);
        AppendEvaluationSummariesMarkdown(builder, report);
        AppendCheckpointMarkdown(builder, report);
        builder.AppendLine("## Loss History");
        builder.AppendLine();
        builder.AppendLine("| Epoch | Loss |");
        builder.AppendLine("| ---: | ---: |");
        for (var index = 0; index < report.LossHistory.Count; index++)
        {
            builder.AppendLine($"| {index + 1} | {report.LossHistory[index].ToString("0.####", CultureInfo.InvariantCulture)} |");
        }

        return builder.ToString().TrimEnd();
    }

    private static string FormatJsonReport(TrainingCommandOptions options, TrainingReport report)
    {
        var document = new TrainingCommandReportDocument(options, report);
        return JsonSerializer.Serialize(document, JsonOptions);
    }

    private static string FormatEvaluationSchedule(TrainingCommandOptions options)
    {
        if (!options.HasEvaluationSchedule)
        {
            return "disabled";
        }

        var cadence = options.EvaluateEvery is null ? "manual" : $"every {options.EvaluateEvery} epoch(s)";
        var dataset = string.IsNullOrWhiteSpace(options.EvaluationDataset)
            ? "default evaluation set"
            : options.EvaluationDataset;
        return $"{cadence} using {dataset}";
    }

    private static string FormatCheckpointSchedule(TrainingCommandOptions options)
    {
        if (!options.HasCheckpointSchedule)
        {
            return "disabled";
        }

        if (options.CheckpointEvery is null && string.IsNullOrWhiteSpace(options.CheckpointDirectory))
        {
            return $"final checkpoint only to current directory (prefix {options.CheckpointPrefix})";
        }

        var cadence = options.CheckpointEvery is null ? "manual" : $"every {options.CheckpointEvery} epoch(s)";
        var target = string.IsNullOrWhiteSpace(options.CheckpointDirectory)
            ? "current directory"
            : options.CheckpointDirectory;
        return $"{cadence} to {target} (prefix {options.CheckpointPrefix})";
    }

    private static string FormatReportSchedule(TrainingCommandOptions options)
    {
        if (!options.HasReportPath)
        {
            return "console summary only";
        }

        return $"{options.ReportPath} ({FormatReportFormat(options.ReportFormat)})";
    }

    private static string FormatReportFormat(TrainingCommandReportFormat format) => format switch
    {
        TrainingCommandReportFormat.Markdown => "markdown",
        TrainingCommandReportFormat.Json => "json",
        _ => "text"
    };

    private static string FormatLossHistory(IReadOnlyList<double> lossHistory) =>
        lossHistory.Count == 0
            ? "(none)"
            : string.Join(", ", lossHistory.Select(loss => loss.ToString("0.####", CultureInfo.InvariantCulture)));

    private static string FormatNullableText(string? value, string? fallback) =>
        string.IsNullOrWhiteSpace(value)
            ? string.IsNullOrWhiteSpace(fallback) ? "(not set)" : fallback!
            : value;

    private static string FormatNullableNumber(double? value) =>
        value is null ? "(n/a)" : value.Value.ToString("0.####", CultureInfo.InvariantCulture);

    private static void AppendEpochMetricsMarkdown(StringBuilder builder, TrainingReport report)
    {
        if (report.EpochMetrics is null || report.EpochMetrics.Count == 0)
        {
            return;
        }

        builder.AppendLine("## Epoch Metrics");
        builder.AppendLine();
        builder.AppendLine("| Epoch | Samples | Tokens | Average loss | Validation perplexity |");
        builder.AppendLine("| ---: | ---: | ---: | ---: | ---: |");
        foreach (var metric in report.EpochMetrics)
        {
            builder.AppendLine(
                $"| {metric.Epoch} | {metric.SamplesSeen} | {metric.TokensSeen} | {metric.AverageLoss.ToString("0.####", CultureInfo.InvariantCulture)} | {FormatNullableNumber(metric.ValidationPerplexity)} |");
        }

        builder.AppendLine();
    }

    private static void AppendEvaluationSummariesMarkdown(StringBuilder builder, TrainingReport report)
    {
        if (report.EvaluationSummaries is null || report.EvaluationSummaries.Count == 0)
        {
            return;
        }

        builder.AppendLine("## Evaluation Summaries");
        builder.AppendLine();
        builder.AppendLine("| Dataset | Samples | Cross entropy | Perplexity |");
        builder.AppendLine("| --- | ---: | ---: | ---: |");
        foreach (var summary in report.EvaluationSummaries)
        {
            builder.AppendLine(
                $"| {summary.Dataset} | {summary.Samples} | {summary.AverageCrossEntropy.ToString("0.####", CultureInfo.InvariantCulture)} | {summary.Perplexity.ToString("0.####", CultureInfo.InvariantCulture)} |");
        }

        builder.AppendLine();
    }

    private static void AppendCheckpointMarkdown(StringBuilder builder, TrainingReport report)
    {
        if (report.Checkpoints is null || report.Checkpoints.Count == 0)
        {
            return;
        }

        builder.AppendLine("## Checkpoints");
        builder.AppendLine();
        builder.AppendLine("| Epoch | Samples seen | Path |");
        builder.AppendLine("| ---: | ---: | --- |");
        foreach (var checkpoint in report.Checkpoints)
        {
            builder.AppendLine($"| {checkpoint.Epoch} | {checkpoint.SamplesSeen} | {checkpoint.Path} |");
        }

        builder.AppendLine();
    }

    private static void AppendEpochMetricsPlainText(StringBuilder builder, TrainingReport report)
    {
        if (report.EpochMetrics is null || report.EpochMetrics.Count == 0)
        {
            return;
        }

        builder.AppendLine();
        builder.AppendLine();
        builder.AppendLine("Epoch metrics:");
        foreach (var metric in report.EpochMetrics)
        {
            builder.AppendLine(
                $"  Epoch {metric.Epoch}: loss={metric.AverageLoss.ToString("0.####", CultureInfo.InvariantCulture)}, samples={metric.SamplesSeen}, tokens={metric.TokensSeen}, validation perplexity={FormatNullableNumber(metric.ValidationPerplexity)}");
        }
    }

    private static void AppendEvaluationSummariesPlainText(StringBuilder builder, TrainingReport report)
    {
        if (report.EvaluationSummaries is null || report.EvaluationSummaries.Count == 0)
        {
            return;
        }

        builder.AppendLine();
        builder.AppendLine();
        builder.AppendLine("Evaluation summaries:");
        foreach (var summary in report.EvaluationSummaries)
        {
            builder.AppendLine(
                $"  {summary.Dataset}: samples={summary.Samples}, cross-entropy={summary.AverageCrossEntropy.ToString("0.####", CultureInfo.InvariantCulture)}, perplexity={summary.Perplexity.ToString("0.####", CultureInfo.InvariantCulture)}");
        }
    }

    private static void AppendCheckpointPlainText(StringBuilder builder, TrainingReport report)
    {
        if (report.Checkpoints is null || report.Checkpoints.Count == 0)
        {
            return;
        }

        builder.AppendLine();
        builder.AppendLine();
        builder.AppendLine("Checkpoints:");
        foreach (var checkpoint in report.Checkpoints)
        {
            builder.AppendLine(
                $"  Epoch {checkpoint.Epoch}: samples={checkpoint.SamplesSeen}, path={checkpoint.Path}");
        }
    }

    private sealed record TrainingCommandReportDocument(
        TrainingCommandOptions Options,
        TrainingReport Report);
}
