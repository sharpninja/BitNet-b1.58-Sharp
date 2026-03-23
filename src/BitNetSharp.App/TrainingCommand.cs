using BitNetSharp.Core;

namespace BitNetSharp.App;

public sealed record TrainingCommandResult(
    TrainingCommandOptions Options,
    TrainingReport? Report,
    string? ReportPath,
    int ExitCode);

public static class TrainingCommand
{
    public static Task<TrainingCommandResult> RunAsync(
        string[] args,
        Func<TrainingCommandOptions, TrainingReport> trainingRunner,
        TextWriter? output = null,
        CancellationToken cancellationToken = default) =>
        RunAsync(args, (options, _) => Task.FromResult(trainingRunner(options)), output, cancellationToken);

    public static async Task<TrainingCommandResult> RunAsync(
        string[] args,
        Func<TrainingCommandOptions, CancellationToken, Task<TrainingReport>> trainingRunner,
        TextWriter? output = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(trainingRunner);

        output ??= Console.Out;

        var options = TrainingCommandOptions.Parse(args);
        if (options.HelpRequested)
        {
            await output.WriteLineAsync(TrainingCommandResultFormatter.FormatUsage());
            return new TrainingCommandResult(options, null, null, 0);
        }

        if (options.DryRun)
        {
            await output.WriteLineAsync(TrainingCommandResultFormatter.FormatPlan(options));
            return new TrainingCommandResult(options, null, null, 0);
        }

        var report = await trainingRunner(options, cancellationToken);
        await output.WriteLineAsync(TrainingCommandResultFormatter.FormatConsoleSummary(options, report));

        string? writtenReportPath = null;
        if (options.HasReportPath)
        {
            writtenReportPath = await WriteReportAsync(options, report, cancellationToken);
            await output.WriteLineAsync(
                TrainingCommandResultFormatter.FormatReportWrittenMessage(writtenReportPath!, options.ReportFormat));
        }

        return new TrainingCommandResult(options, report, writtenReportPath, 0);
    }

    private static async Task<string> WriteReportAsync(
        TrainingCommandOptions options,
        TrainingReport report,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(report);

        if (!options.HasReportPath)
        {
            throw new InvalidOperationException("A report path is required before writing the training report.");
        }

        var reportPath = options.ReportPath!;
        var directory = Path.GetDirectoryName(reportPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var reportDocument = TrainingCommandResultFormatter.FormatReportDocument(options, report);
        await File.WriteAllTextAsync(reportPath, reportDocument, cancellationToken);
        return reportPath;
    }
}
