using System.Net;
using System.Text;
using System.Text.Json;
using BitNetSharp.Core;

namespace BitNetSharp.App;

public sealed record HostedAgentBenchmarkQueryResult(
    string Prompt,
    string ExpectedResponse,
    string ActualResponse,
    bool ResponseSucceeded,
    bool ExactMatch,
    double ExpectedTokenRecall);

public sealed record HostedAgentBenchmarkModelReport(
    string ModelSpecifier,
    string DisplayName,
    bool TrainingSupported,
    bool TrainingCompleted,
    int TrainingExamples,
    int TrainingEpochs,
    int SuccessfulQueries,
    int TotalQueries,
    int ExactMatches,
    double AverageExpectedTokenRecall,
    IReadOnlyList<HostedAgentBenchmarkQueryResult> QueryResults,
    BitNetPaperAuditReport? PaperAlignmentAudit = null,
    double? WikiText2Perplexity = null,
    int BenchmarkPromptTokenCount = 0)
{
    public double EfficacyRate => TotalQueries == 0 ? 0d : SuccessfulQueries / (double)TotalQueries;

    public double ExactMatchRate => TotalQueries == 0 ? 0d : ExactMatches / (double)TotalQueries;
}

public sealed record HostedAgentBenchmarkComparisonMetric(
    string ModelSpecifier,
    string DisplayName,
    string? ResponseMean,
    string? TrainingMean,
    double? ResponseTokensPerSecond,
    double? ResponseAllocatedMegabytes,
    double? WikiText2Perplexity);

public sealed record HostedAgentBenchmarkPerformanceRow(
    string Operation,
    string ModelSpecifier,
    string Mean,
    string StdDev,
    string Allocated,
    string HtmlReportPath,
    string CsvReportPath,
    string MarkdownReportPath);

public sealed record HostedAgentBenchmarkComparisonReport(
    DateTimeOffset GeneratedAtUtc,
    IReadOnlyList<string> QueryScript,
    IReadOnlyList<HostedAgentBenchmarkModelReport> Models,
    IReadOnlyList<HostedAgentBenchmarkPerformanceRow> PerformanceRows,
    HostedAgentBenchmarkComparisonSummary? ComparisonSummary = null);

public sealed record HostedAgentBenchmarkComparisonSummary(
    string PerplexityDataset,
    IReadOnlyList<HostedAgentBenchmarkComparisonMetric> Models,
    double? BitNetSpeedupVersusTraditional,
    double? BitNetMemoryDeltaPercentVersusTraditional,
    double? BitNetQualityDeltaPercentVersusTraditional);

public static class HostedAgentBenchmarkReportRunner
{
    private const string StampedJsonTimestampFormat = "yyyyMMddTHHmmssZ";
    private const string ResponseOperation = "SpecFlow: Generate a response for a prompt";
    private const string TrainingOperation = "SpecFlow: Train the selected model on the default dataset";
    private const double NanosecondsPerMillisecond = 1_000_000d;
    private const double MicrosecondsPerMillisecond = 1_000d;
    private const double MillisecondsPerSecond = 1_000d;
    private const double BytesPerMegabyte = 1024d * 1024d;
    private const double KilobytesPerMegabyte = 1024d;
    public static async Task<string> RunAsync(
        HostedAgentBenchmarkOptions options,
        string? outputDirectory,
        string? commitHash = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        var originalWorkingDirectory = Directory.GetCurrentDirectory();
        var reportDirectory = Path.GetFullPath(
            string.IsNullOrWhiteSpace(outputDirectory)
                ? Path.Combine(originalWorkingDirectory, "artifacts", "benchmark-report")
                : outputDirectory);
        Directory.CreateDirectory(reportDirectory);
        HostedAgentBenchmarkRunner.Run(options);
        CopyArtifactsDirectory(
            Path.Combine(originalWorkingDirectory, "BenchmarkDotNet.Artifacts"),
            Path.Combine(reportDirectory, "BenchmarkDotNet.Artifacts"));

        var trainingExamples = BitNetTrainingCorpus.CreateDefaultExamples();
        var modelReports = await CreateModelReportsAsync(options, trainingExamples, cancellationToken);
        var performanceRows = ParsePerformanceRows(reportDirectory);
        var comparisonSummary = CreateComparisonSummary(modelReports, performanceRows);
        var report = new HostedAgentBenchmarkComparisonReport(
            DateTimeOffset.UtcNow,
            trainingExamples.Select(static example => example.Prompt).ToArray(),
            modelReports,
            performanceRows,
            comparisonSummary);

        WriteReportSite(reportDirectory, report, commitHash);
        return reportDirectory;
    }

    public static IReadOnlyList<HostedAgentBenchmarkPerformanceRow> ParsePerformanceRows(string reportDirectory)
    {
        var resultsDirectory = Path.Combine(reportDirectory, "BenchmarkDotNet.Artifacts", "results");
        if (!Directory.Exists(resultsDirectory))
        {
            return [];
        }

        var rows = new List<HostedAgentBenchmarkPerformanceRow>();
        foreach (var markdownPath in Directory.GetFiles(resultsDirectory, "*-report-github.md", SearchOption.TopDirectoryOnly).OrderBy(static path => path, StringComparer.Ordinal))
        {
            var reportFileName = Path.GetFileName(markdownPath);
            var reportStem = reportFileName[..^"-report-github.md".Length];
            var htmlPath = Path.Combine(resultsDirectory, $"{reportStem}-report.html");
            var csvPath = Path.Combine(resultsDirectory, $"{reportStem}-report.csv");

            var headerLine = File.ReadLines(markdownPath)
                .FirstOrDefault(static line => line.StartsWith("| Method", StringComparison.Ordinal));
            if (headerLine is null)
            {
                continue;
            }

            var headers = SplitTableRow(headerLine);
            foreach (var line in File.ReadLines(markdownPath).Where(static line => line.StartsWith("|", StringComparison.Ordinal)))
            {
                var cells = SplitTableRow(line);
                if (cells.Count != headers.Count || cells.Count == 0)
                {
                    continue;
                }

                if (cells.All(static cell => cell.All(static character => character is '-' or ':')))
                {
                    continue;
                }

                if (string.Equals(cells[0], "Method", StringComparison.Ordinal))
                {
                    continue;
                }

                var values = headers.Zip(cells).ToDictionary(static pair => pair.First, static pair => pair.Second, StringComparer.Ordinal);
                if (!values.TryGetValue("Method", out var operation)
                    || !values.TryGetValue("ModelSpecifier", out var modelSpecifier)
                    || !values.TryGetValue("Mean", out var mean)
                    || !values.TryGetValue("StdDev", out var stdDev))
                {
                    continue;
                }

                values.TryGetValue("Allocated", out var allocated);
                rows.Add(new HostedAgentBenchmarkPerformanceRow(
                    operation,
                    modelSpecifier,
                    mean,
                    stdDev,
                    string.IsNullOrWhiteSpace(allocated) ? "-" : allocated,
                    ToRelativeUnixPath(reportDirectory, htmlPath),
                    ToRelativeUnixPath(reportDirectory, csvPath),
                    ToRelativeUnixPath(reportDirectory, markdownPath)));
            }
        }

        return rows
            .OrderBy(static row => row.Operation, StringComparer.Ordinal)
            .ThenBy(static row => row.ModelSpecifier, StringComparer.Ordinal)
            .ToArray();
    }

    public static void WriteReportSite(string outputDirectory, HostedAgentBenchmarkComparisonReport report, string? commitHash = null)
    {
        ArgumentNullException.ThrowIfNull(report);

        Directory.CreateDirectory(outputDirectory);
        File.WriteAllText(Path.Combine(outputDirectory, "comparison-report.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(outputDirectory, "comparison-report.md"), BuildMarkdown(report));
        File.WriteAllText(Path.Combine(outputDirectory, "index.html"), BuildHtml(report, commitHash));

        if (!string.IsNullOrWhiteSpace(commitHash))
        {
            var timestamp = report.GeneratedAtUtc.ToString(StampedJsonTimestampFormat);
            var stampedJsonFileName = $"comparison-report-{commitHash}-{timestamp}.json";
            File.WriteAllText(
                Path.Combine(outputDirectory, stampedJsonFileName),
                JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        }
    }

    private static async Task<IReadOnlyList<HostedAgentBenchmarkModelReport>> CreateModelReportsAsync(
        HostedAgentBenchmarkOptions options,
        IReadOnlyList<TrainingExample> trainingExamples,
        CancellationToken cancellationToken)
    {
        var reports = new List<HostedAgentBenchmarkModelReport>();
        foreach (var modelSpecifier in options.ModelSpecifiers)
        {
            using var model = HostedAgentModelFactory.Create(modelSpecifier, options.Verbosity);
            var trainingSupported = model is ITrainableHostedAgentModel;
            var trainingCompleted = false;
            var trainingEpochs = 0;

            if (trainingSupported)
            {
                trainingEpochs = GetTrainingEpochs(model);
                ((ITrainableHostedAgentModel)model).Train(trainingExamples, trainingEpochs);
                trainingCompleted = true;
            }

            var queryResults = new List<HostedAgentBenchmarkQueryResult>(trainingExamples.Count);
            foreach (var example in trainingExamples)
            {
                var response = await model.GetResponseAsync(example.Prompt, options.MaxOutputTokens, cancellationToken);
                var exactMatch = Normalize(response.Text) == Normalize(example.Response);
                queryResults.Add(new HostedAgentBenchmarkQueryResult(
                    example.Prompt,
                    example.Response,
                    response.Text,
                    !string.IsNullOrWhiteSpace(response.Text),
                    exactMatch,
                    CalculateExpectedTokenRecall(response.Text, example.Response)));
            }

            reports.Add(new HostedAgentBenchmarkModelReport(
                model.ModelId,
                model.DisplayName,
                trainingSupported,
                trainingCompleted,
                trainingCompleted ? trainingExamples.Count : 0,
                trainingEpochs,
                queryResults.Count(static result => result.ResponseSucceeded),
                queryResults.Count,
                queryResults.Count(static result => result.ExactMatch),
                queryResults.Count == 0 ? 0d : queryResults.Average(static result => result.ExpectedTokenRecall),
                queryResults,
                model is BitNetHostedAgentModel bitNetModel ? BitNetPaperAuditor.CreateReport(bitNetModel.Model) : null,
                GetWikiText2Perplexity(model),
                await GetBenchmarkPromptTokenCountAsync(model, options, cancellationToken)));
        }

        return reports;
    }

    private static HostedAgentBenchmarkComparisonSummary? CreateComparisonSummary(
        IReadOnlyList<HostedAgentBenchmarkModelReport> modelReports,
        IReadOnlyList<HostedAgentBenchmarkPerformanceRow> performanceRows)
    {
        if (modelReports.Count == 0)
        {
            return null;
        }

        var metrics = modelReports
            .Select(modelReport =>
            {
                var responseRow = performanceRows.FirstOrDefault(row =>
                    string.Equals(row.Operation, ResponseOperation, StringComparison.Ordinal)
                    && string.Equals(row.ModelSpecifier, modelReport.ModelSpecifier, StringComparison.Ordinal));
                var trainingRow = performanceRows.FirstOrDefault(row =>
                    string.Equals(row.Operation, TrainingOperation, StringComparison.Ordinal)
                    && string.Equals(row.ModelSpecifier, modelReport.ModelSpecifier, StringComparison.Ordinal));
                var responseMeanMilliseconds = TryParseDurationMilliseconds(responseRow?.Mean);
                double? responseTokensPerSecond = responseMeanMilliseconds is > 0d && modelReport.BenchmarkPromptTokenCount > 0
                    ? (modelReport.BenchmarkPromptTokenCount * MillisecondsPerSecond) / responseMeanMilliseconds.Value
                    : null;

                return new HostedAgentBenchmarkComparisonMetric(
                    modelReport.ModelSpecifier,
                    modelReport.DisplayName,
                    responseRow?.Mean,
                    trainingRow?.Mean,
                    responseTokensPerSecond,
                    TryParseAllocatedMegabytes(responseRow?.Allocated),
                    modelReport.WikiText2Perplexity);
            })
            .ToArray();

        var bitNetMetric = metrics.FirstOrDefault(metric =>
            string.Equals(metric.ModelSpecifier, HostedAgentModelFactory.DefaultModelId, StringComparison.Ordinal));
        var traditionalMetric = metrics.FirstOrDefault(metric =>
            string.Equals(metric.ModelSpecifier, HostedAgentModelFactory.TraditionalLocalModelId, StringComparison.Ordinal));

        return new HostedAgentBenchmarkComparisonSummary(
            "WikiText2",
            metrics,
            bitNetMetric?.ResponseTokensPerSecond is > 0d && traditionalMetric?.ResponseTokensPerSecond is > 0d
                ? bitNetMetric.ResponseTokensPerSecond / traditionalMetric.ResponseTokensPerSecond
                : null,
            bitNetMetric?.ResponseAllocatedMegabytes is > 0d && traditionalMetric?.ResponseAllocatedMegabytes is > 0d
                ? ((traditionalMetric.ResponseAllocatedMegabytes.Value - bitNetMetric.ResponseAllocatedMegabytes.Value)
                    / traditionalMetric.ResponseAllocatedMegabytes.Value) * 100d
                : null,
            bitNetMetric?.WikiText2Perplexity is > 0d && traditionalMetric?.WikiText2Perplexity is > 0d
                ? ((traditionalMetric.WikiText2Perplexity.Value - bitNetMetric.WikiText2Perplexity.Value)
                    / traditionalMetric.WikiText2Perplexity.Value) * 100d
                : null);
    }

    private static string BuildMarkdown(HostedAgentBenchmarkComparisonReport report)
    {
        var builder = new StringBuilder();
        builder.AppendLine("# BitNet benchmark comparison report");
        builder.AppendLine();
        builder.AppendLine($"Generated: {report.GeneratedAtUtc:O}");
        builder.AppendLine();
        builder.AppendLine("## Shared integration inputs");
        builder.AppendLine();
        builder.AppendLine("- Training set: `BitNetTrainingCorpus.CreateDefaultExamples()`");
        builder.AppendLine("- Query script:");
        foreach (var prompt in report.QueryScript)
        {
            builder.AppendLine($"  - `{prompt}`");
        }

        builder.AppendLine();
        builder.AppendLine("## Efficacy and accuracy summary");
        builder.AppendLine();
        builder.AppendLine("| Model | Training | Efficacy | Exact-match accuracy | Expected-token recall |");
        builder.AppendLine("| --- | --- | ---: | ---: | ---: |");
        foreach (var model in report.Models)
        {
            builder.AppendLine(
                $"| {model.ModelSpecifier} | {FormatTrainingStatus(model)} | {model.EfficacyRate:P1} | {model.ExactMatchRate:P1} | {model.AverageExpectedTokenRecall:P1} |");
        }

        builder.AppendLine();
        AppendComparisonSummaryMarkdown(builder, report.ComparisonSummary);
        builder.AppendLine();
        builder.AppendLine("## Query script results");
        builder.AppendLine();
        builder.AppendLine("| Model | Prompt | Expected-token recall | Exact match | Response |");
        builder.AppendLine("| --- | --- | ---: | --- | --- |");
        foreach (var model in report.Models)
        {
            foreach (var query in model.QueryResults)
            {
                builder.AppendLine(
                    $"| {model.ModelSpecifier} | `{query.Prompt}` | {query.ExpectedTokenRecall:P1} | {(query.ExactMatch ? "Yes" : "No")} | {EscapeMarkdownInline(query.ActualResponse)} |");
            }
        }

        builder.AppendLine();
        AppendPaperAuditMarkdown(builder, report);
        builder.AppendLine();
        builder.AppendLine("## BenchmarkDotNet performance summary");
        builder.AppendLine();
        builder.AppendLine("| Operation | Model | Mean | StdDev | Allocated | Reports |");
        builder.AppendLine("| --- | --- | ---: | ---: | ---: | --- |");
        foreach (var row in report.PerformanceRows)
        {
            builder.AppendLine(
                $"| {row.Operation} | {row.ModelSpecifier} | {row.Mean} | {row.StdDev} | {row.Allocated} | [HTML]({row.HtmlReportPath}) · [CSV]({row.CsvReportPath}) · [Markdown]({row.MarkdownReportPath}) |");
        }

        return builder.ToString();
    }

    private static string BuildHtml(HostedAgentBenchmarkComparisonReport report, string? commitHash = null)
    {
        var builder = new StringBuilder();
        builder.AppendLine("<!DOCTYPE html>");
        builder.AppendLine("<html lang=\"en\">");
        builder.AppendLine("<head>");
        builder.AppendLine("  <meta charset=\"utf-8\" />");
        builder.AppendLine("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />");
        builder.AppendLine("  <title>BitNet benchmark comparison report</title>");
        builder.AppendLine("  <style>");
        builder.AppendLine("    body { font-family: Arial, Helvetica, sans-serif; margin: 2rem auto; max-width: 1200px; color: #1f2937; line-height: 1.5; padding: 0 1rem; }");
        builder.AppendLine("    h1, h2 { color: #111827; }");
        builder.AppendLine("    table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }");
        builder.AppendLine("    th, td { border: 1px solid #d1d5db; padding: 0.75rem; text-align: left; vertical-align: top; }");
        builder.AppendLine("    th { background: #f3f4f6; }");
        builder.AppendLine("    code { background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 0.25rem; }");
        builder.AppendLine("    ul { padding-left: 1.25rem; }");
        builder.AppendLine("    .muted { color: #6b7280; }");
        builder.AppendLine("  </style>");
        builder.AppendLine("</head>");
        builder.AppendLine("<body>");
        builder.AppendLine("  <h1>BitNet benchmark comparison report</h1>");
        builder.AppendLine($"  <p class=\"muted\">Generated {Encode(report.GeneratedAtUtc.ToString("O"))}</p>");
        builder.AppendLine("  <h2>Shared integration inputs</h2>");
        builder.AppendLine("  <ul>");
        builder.AppendLine("    <li>Training set: <code>BitNetTrainingCorpus.CreateDefaultExamples()</code></li>");
        builder.AppendLine("    <li>Query script:");
        builder.AppendLine("      <ul>");
        foreach (var prompt in report.QueryScript)
        {
            builder.AppendLine($"        <li><code>{Encode(prompt)}</code></li>");
        }

        builder.AppendLine("      </ul>");
        builder.AppendLine("    </li>");
        builder.AppendLine("  </ul>");

        builder.AppendLine("  <h2>Efficacy and accuracy summary</h2>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Model</th><th>Training</th><th>Efficacy</th><th>Exact-match accuracy</th><th>Expected-token recall</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in report.Models)
        {
            builder.AppendLine($"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{Encode(FormatTrainingStatus(model))}</td><td>{Encode(model.EfficacyRate.ToString("P1"))}</td><td>{Encode(model.ExactMatchRate.ToString("P1"))}</td><td>{Encode(model.AverageExpectedTokenRecall.ToString("P1"))}</td></tr>");
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        AppendComparisonSummaryHtml(builder, report.ComparisonSummary);

        builder.AppendLine("  <h2>Query script results</h2>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Model</th><th>Prompt</th><th>Expected-token recall</th><th>Exact match</th><th>Response</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in report.Models)
        {
            foreach (var query in model.QueryResults)
            {
                builder.AppendLine($"      <tr><td>{Encode(model.ModelSpecifier)}</td><td><code>{Encode(query.Prompt)}</code></td><td>{Encode(query.ExpectedTokenRecall.ToString("P1"))}</td><td>{(query.ExactMatch ? "Yes" : "No")}</td><td>{Encode(query.ActualResponse)}</td></tr>");
            }
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        AppendPaperAuditHtml(builder, report);

        builder.AppendLine("  <h2>BenchmarkDotNet performance summary</h2>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Operation</th><th>Model</th><th>Mean</th><th>StdDev</th><th>Allocated</th><th>Reports</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var row in report.PerformanceRows)
        {
            builder.AppendLine(
                $"      <tr><td>{Encode(row.Operation)}</td><td>{Encode(row.ModelSpecifier)}</td><td>{Encode(row.Mean)}</td><td>{Encode(row.StdDev)}</td><td>{Encode(row.Allocated)}</td><td><a href=\"{Encode(row.HtmlReportPath)}\">HTML</a> · <a href=\"{Encode(row.CsvReportPath)}\">CSV</a> · <a href=\"{Encode(row.MarkdownReportPath)}\">Markdown</a></td></tr>");
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        var stampedJsonLink = !string.IsNullOrWhiteSpace(commitHash)
            ? $" · <a href=\"comparison-report-{Encode(commitHash)}-{Encode(report.GeneratedAtUtc.ToString(StampedJsonTimestampFormat))}.json\">Download the stamped JSON report</a>"
            : string.Empty;
        builder.AppendLine($"  <p><a href=\"comparison-report.md\">Download the Markdown report</a> · <a href=\"comparison-report.json\">Download the JSON report</a>{stampedJsonLink}</p>");
        builder.AppendLine("</body>");
        builder.AppendLine("</html>");
        return builder.ToString();
    }

    private static double CalculateExpectedTokenRecall(string actualResponse, string expectedResponse)
    {
        var expectedTokens = Tokenize(expectedResponse);
        if (expectedTokens.Length == 0)
        {
            return 0d;
        }

        var actualTokens = Tokenize(actualResponse).ToHashSet(StringComparer.Ordinal);
        var matched = expectedTokens.Count(token => actualTokens.Contains(token));
        return matched / (double)expectedTokens.Length;
    }

    private static string[] Tokenize(string value) =>
        value.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(static token => NormalizeToken(token))
            .Where(static token => !string.IsNullOrWhiteSpace(token))
            .ToArray();

    private static string Normalize(string value) =>
        string.Join(' ', Tokenize(value));

    private static string NormalizeToken(string token) =>
        token.Trim().ToLowerInvariant().Trim(',', '.', '!', '?', ';', ':', '"', '\'');

    private static int GetTrainingEpochs(IHostedAgentModel model) =>
        string.Equals(model.ModelId, HostedAgentModelFactory.TraditionalLocalModelId, StringComparison.Ordinal)
            ? TraditionalLocalModel.DefaultTrainingEpochs
            : 3;

    private static IReadOnlyList<string> SplitTableRow(string line) =>
        line.Trim()
            .Trim('|')
            .Split('|')
            .Select(static cell => WebUtility.HtmlDecode(cell).Replace("**", string.Empty, StringComparison.Ordinal).Trim().Trim('\''))
            .ToArray();

    private static string ToRelativeUnixPath(string rootDirectory, string path) =>
        Path.GetRelativePath(rootDirectory, path).Replace(Path.DirectorySeparatorChar, '/');

    private static void CopyArtifactsDirectory(string sourceDirectory, string destinationDirectory)
    {
        if (!Directory.Exists(sourceDirectory))
        {
            return;
        }

        if (Directory.Exists(destinationDirectory))
        {
            Directory.Delete(destinationDirectory, recursive: true);
        }

        Directory.CreateDirectory(destinationDirectory);
        foreach (var directory in Directory.GetDirectories(sourceDirectory, "*", SearchOption.AllDirectories))
        {
            Directory.CreateDirectory(directory.Replace(sourceDirectory, destinationDirectory, StringComparison.Ordinal));
        }

        foreach (var file in Directory.GetFiles(sourceDirectory, "*", SearchOption.AllDirectories))
        {
            var destinationFile = file.Replace(sourceDirectory, destinationDirectory, StringComparison.Ordinal);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationFile)!);
            File.Copy(file, destinationFile, overwrite: true);
        }
    }

    private static string FormatTrainingStatus(HostedAgentBenchmarkModelReport model) =>
        model.TrainingSupported && model.TrainingCompleted
            ? $"Completed ({model.TrainingExamples} examples, {model.TrainingEpochs} epochs)"
            : "Not supported";

    private static async Task<int> GetBenchmarkPromptTokenCountAsync(
        IHostedAgentModel model,
        HostedAgentBenchmarkOptions options,
        CancellationToken cancellationToken)
    {
        var benchmarkResponse = await model.GetResponseAsync(options.Prompt, options.MaxOutputTokens, cancellationToken);
        return CountResponseTokens(model, benchmarkResponse.Text);
    }

    private static double? GetWikiText2Perplexity(IHostedAgentModel model) =>
        model switch
        {
            BitNetHostedAgentModel bitNetModel => bitNetModel.Model.CalculatePerplexity(BitNetBenchmarkFixtures.WikiText2ValidationSamples),
            TraditionalLocalHostedAgentModel traditionalModel => traditionalModel.Model.CalculatePerplexity(BitNetBenchmarkFixtures.WikiText2ValidationSamples),
            _ => null
        };

    private static int CountResponseTokens(IHostedAgentModel model, string responseText) =>
        model switch
        {
            BitNetHostedAgentModel bitNetModel => bitNetModel.Model.Tokenizer.Tokenize(responseText).Count(),
            TraditionalLocalHostedAgentModel traditionalModel => traditionalModel.Model.Tokenizer.Tokenize(responseText).Count(),
            _ => Tokenize(responseText).Length
        };

    private static double? TryParseDurationMilliseconds(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        var parts = value.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 2 || !double.TryParse(parts[0], out var magnitude))
        {
            return null;
        }

        return parts[1] switch
        {
            "ns" => magnitude / NanosecondsPerMillisecond,
            "μs" or "us" => magnitude / MicrosecondsPerMillisecond,
            "ms" => magnitude,
            "s" => magnitude * MillisecondsPerSecond,
            _ => null
        };
    }

    private static double? TryParseAllocatedMegabytes(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || string.Equals(value, "-", StringComparison.Ordinal))
        {
            return null;
        }

        var parts = value.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 2 || !double.TryParse(parts[0], out var magnitude))
        {
            return null;
        }

        return parts[1] switch
        {
            "B" => magnitude / BytesPerMegabyte,
            "KB" => magnitude / KilobytesPerMegabyte,
            "MB" => magnitude,
            "GB" => magnitude * KilobytesPerMegabyte,
            _ => null
        };
    }

    private static void AppendComparisonSummaryMarkdown(StringBuilder builder, HostedAgentBenchmarkComparisonSummary? summary)
    {
        if (summary is null)
        {
            return;
        }

        builder.AppendLine("## BitNet vs traditional comparison summary");
        builder.AppendLine();
        builder.AppendLine($"Perplexity dataset: `{summary.PerplexityDataset}`");
        builder.AppendLine();
        builder.AppendLine("| Model | Response mean | Response tokens/sec | Training mean | Perplexity | Response allocated |");
        builder.AppendLine("| --- | ---: | ---: | ---: | ---: | ---: |");
        foreach (var model in summary.Models)
        {
            builder.AppendLine(
                $"| {model.ModelSpecifier} | {model.ResponseMean ?? "-"} | {FormatNullableNumber(model.ResponseTokensPerSecond)} | {model.TrainingMean ?? "-"} | {FormatNullableNumber(model.WikiText2Perplexity)} | {FormatNullableMegabytes(model.ResponseAllocatedMegabytes)} |");
        }

        builder.AppendLine();
        builder.AppendLine("| Delta | Value |");
        builder.AppendLine("| --- | ---: |");
        builder.AppendLine($"| BitNet speedup vs traditional | {FormatNullableRatio(summary.BitNetSpeedupVersusTraditional)} |");
        builder.AppendLine($"| BitNet memory reduction vs traditional | {FormatNullablePercent(summary.BitNetMemoryDeltaPercentVersusTraditional)} |");
        builder.AppendLine($"| BitNet quality improvement vs traditional | {FormatNullablePercent(summary.BitNetQualityDeltaPercentVersusTraditional)} |");
    }

    private static void AppendComparisonSummaryHtml(StringBuilder builder, HostedAgentBenchmarkComparisonSummary? summary)
    {
        if (summary is null)
        {
            return;
        }

        builder.AppendLine("  <h2>BitNet vs traditional comparison summary</h2>");
        builder.AppendLine($"  <p>Perplexity dataset: <code>{Encode(summary.PerplexityDataset)}</code></p>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Model</th><th>Response mean</th><th>Response tokens/sec</th><th>Training mean</th><th>Perplexity</th><th>Response allocated</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in summary.Models)
        {
            builder.AppendLine(
                $"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{Encode(model.ResponseMean ?? "-")}</td><td>{Encode(FormatNullableNumber(model.ResponseTokensPerSecond))}</td><td>{Encode(model.TrainingMean ?? "-")}</td><td>{Encode(FormatNullableNumber(model.WikiText2Perplexity))}</td><td>{Encode(FormatNullableMegabytes(model.ResponseAllocatedMegabytes))}</td></tr>");
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Delta</th><th>Value</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        builder.AppendLine($"      <tr><td>BitNet speedup vs traditional</td><td>{Encode(FormatNullableRatio(summary.BitNetSpeedupVersusTraditional))}</td></tr>");
        builder.AppendLine($"      <tr><td>BitNet memory reduction vs traditional</td><td>{Encode(FormatNullablePercent(summary.BitNetMemoryDeltaPercentVersusTraditional))}</td></tr>");
        builder.AppendLine($"      <tr><td>BitNet quality improvement vs traditional</td><td>{Encode(FormatNullablePercent(summary.BitNetQualityDeltaPercentVersusTraditional))}</td></tr>");
        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        AppendComparisonChartsHtml(builder, summary);
    }

    private static void AppendComparisonChartsHtml(StringBuilder builder, HostedAgentBenchmarkComparisonSummary summary)
    {
        var throughputMax = summary.Models.Max(static model => model.ResponseTokensPerSecond ?? 0d);
        var memoryMax = summary.Models.Max(static model => model.ResponseAllocatedMegabytes ?? 0d);
        var perplexityMax = summary.Models.Max(static model => model.WikiText2Perplexity ?? 0d);
        var perplexityMin = summary.Models
            .Where(static model => model.WikiText2Perplexity is > 0d)
            .Select(static model => model.WikiText2Perplexity!.Value)
            .DefaultIfEmpty(0d)
            .Min();

        builder.AppendLine("  <h3>Comparison charts</h3>");
        AppendBarChart(builder, "Response tokens/sec", summary.Models, throughputMax, 0d, static model => model.ResponseTokensPerSecond, FormatNullableNumber);
        AppendBarChart(builder, "Response allocated (MB)", summary.Models, memoryMax, 0d, static model => model.ResponseAllocatedMegabytes, FormatNullableMegabytes);
        AppendBarChart(builder, "Perplexity", summary.Models, perplexityMax, perplexityMin, static model => model.WikiText2Perplexity, FormatNullableNumber, lowerIsBetter: true);
    }

    private static void AppendBarChart(
        StringBuilder builder,
        string title,
        IReadOnlyList<HostedAgentBenchmarkComparisonMetric> models,
        double maxValue,
        double minValue,
        Func<HostedAgentBenchmarkComparisonMetric, double?> selector,
        Func<double?, string> formatter,
        bool lowerIsBetter = false)
    {
        builder.AppendLine($"  <h4>{Encode(title)}</h4>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Model</th><th>Value</th><th>Chart</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in models)
        {
            var value = selector(model);
            var widthPercent = value is > 0d
                ? GetBarWidthPercent(value.Value, minValue, maxValue, lowerIsBetter)
                : 0d;
            builder.AppendLine(
                $"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{Encode(formatter(value))}</td><td><div style=\"background:#e5e7eb;border-radius:9999px;height:0.9rem;min-width:12rem;\"><div style=\"background:#2563eb;border-radius:9999px;height:0.9rem;width:{widthPercent:0.#}%;\"></div></div></td></tr>");
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
    }

    private static double GetBarWidthPercent(double value, double minValue, double maxValue, bool lowerIsBetter)
    {
        if (maxValue <= 0d)
        {
            return 0d;
        }

        if (!lowerIsBetter)
        {
            return Math.Clamp((value / maxValue) * 100d, 0d, 100d);
        }

        if (maxValue <= minValue)
        {
            return 100d;
        }

        return Math.Clamp(((maxValue - value) / (maxValue - minValue)) * 100d, 0d, 100d);
    }

    private static void AppendPaperAuditMarkdown(StringBuilder builder, HostedAgentBenchmarkComparisonReport report)
    {
        var auditedModels = report.Models.Where(static model => model.PaperAlignmentAudit is not null).ToArray();
        if (auditedModels.Length == 0)
        {
            return;
        }

        builder.AppendLine("## Paper-alignment audit");
        builder.AppendLine();
        builder.AppendLine("| Model | Passed | Pending | Failed |");
        builder.AppendLine("| --- | ---: | ---: | ---: |");
        foreach (var model in auditedModels)
        {
            var audit = model.PaperAlignmentAudit!;
            builder.AppendLine($"| {model.ModelSpecifier} | {audit.PassedCount} | {audit.PendingCount} | {audit.FailedCount} |");
        }

        builder.AppendLine();
        builder.AppendLine("| Model | Area | Status | Requirement | Details |");
        builder.AppendLine("| --- | --- | --- | --- | --- |");
        foreach (var model in auditedModels)
        {
            foreach (var check in model.PaperAlignmentAudit!.Checks)
            {
                builder.AppendLine(
                    $"| {model.ModelSpecifier} | {check.Area} | {check.Status} | {EscapeMarkdownInline(check.Requirement)} | {EscapeMarkdownInline(check.Details)} |");
            }
        }
    }

    private static void AppendPaperAuditHtml(StringBuilder builder, HostedAgentBenchmarkComparisonReport report)
    {
        var auditedModels = report.Models.Where(static model => model.PaperAlignmentAudit is not null).ToArray();
        if (auditedModels.Length == 0)
        {
            return;
        }

        builder.AppendLine("  <h2>Paper-alignment audit</h2>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Model</th><th>Passed</th><th>Pending</th><th>Failed</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in auditedModels)
        {
            var audit = model.PaperAlignmentAudit!;
            builder.AppendLine($"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{audit.PassedCount}</td><td>{audit.PendingCount}</td><td>{audit.FailedCount}</td></tr>");
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Model</th><th>Area</th><th>Status</th><th>Requirement</th><th>Details</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in auditedModels)
        {
            foreach (var check in model.PaperAlignmentAudit!.Checks)
            {
                builder.AppendLine($"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{Encode(check.Area)}</td><td>{Encode(check.Status.ToString())}</td><td>{Encode(check.Requirement)}</td><td>{Encode(check.Details)}</td></tr>");
            }
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
    }

    private static string EscapeMarkdownInline(string value) =>
        value.Replace("|", "\\|", StringComparison.Ordinal).Replace(Environment.NewLine, "<br />", StringComparison.Ordinal);

    private static string FormatNullableNumber(double? value) => value is null ? "-" : value.Value.ToString("0.##");

    private static string FormatNullableMegabytes(double? value) => value is null ? "-" : $"{value.Value:0.##} MB";

    private static string FormatNullablePercent(double? value) => value is null ? "-" : $"{value.Value:0.##}%";

    private static string FormatNullableRatio(double? value) => value is null ? "-" : $"{value.Value:0.##}x";

    private static string Encode(string value) => WebUtility.HtmlEncode(value);
}
