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
    IReadOnlyList<HostedAgentBenchmarkQueryResult> QueryResults)
{
    public double EfficacyRate => TotalQueries == 0 ? 0d : SuccessfulQueries / (double)TotalQueries;

    public double ExactMatchRate => TotalQueries == 0 ? 0d : ExactMatches / (double)TotalQueries;
}

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
    IReadOnlyList<HostedAgentBenchmarkPerformanceRow> PerformanceRows);

public static class HostedAgentBenchmarkReportRunner
{
    public static async Task<string> RunAsync(
        HostedAgentBenchmarkOptions options,
        string? outputDirectory,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        var reportDirectory = Path.GetFullPath(
            string.IsNullOrWhiteSpace(outputDirectory)
                ? Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "benchmark-report")
                : outputDirectory);
        Directory.CreateDirectory(reportDirectory);

        var originalWorkingDirectory = Directory.GetCurrentDirectory();
        try
        {
            Directory.SetCurrentDirectory(reportDirectory);
            HostedAgentBenchmarkRunner.Run(options);
        }
        finally
        {
            Directory.SetCurrentDirectory(originalWorkingDirectory);
        }

        var trainingExamples = BitNetTrainingCorpus.CreateDefaultExamples();
        var modelReports = await CreateModelReportsAsync(options, trainingExamples, cancellationToken);
        var performanceRows = ParsePerformanceRows(reportDirectory);
        var report = new HostedAgentBenchmarkComparisonReport(
            DateTimeOffset.UtcNow,
            trainingExamples.Select(static example => example.Prompt).ToArray(),
            modelReports,
            performanceRows);

        WriteReportSite(reportDirectory, report);
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

    public static void WriteReportSite(string outputDirectory, HostedAgentBenchmarkComparisonReport report)
    {
        ArgumentNullException.ThrowIfNull(report);

        Directory.CreateDirectory(outputDirectory);
        File.WriteAllText(Path.Combine(outputDirectory, "comparison-report.json"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(outputDirectory, "comparison-report.md"), BuildMarkdown(report));
        File.WriteAllText(Path.Combine(outputDirectory, "index.html"), BuildHtml(report));
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
                queryResults));
        }

        return reports;
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

    private static string BuildHtml(HostedAgentBenchmarkComparisonReport report)
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
        builder.AppendLine("  <p><a href=\"comparison-report.md\">Download the Markdown report</a> · <a href=\"comparison-report.json\">Download the JSON report</a></p>");
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

    private static string FormatTrainingStatus(HostedAgentBenchmarkModelReport model) =>
        model.TrainingSupported && model.TrainingCompleted
            ? $"Completed ({model.TrainingExamples} examples, {model.TrainingEpochs} epochs)"
            : "Not supported";

    private static string EscapeMarkdownInline(string value) =>
        value.Replace("|", "\\|", StringComparison.Ordinal).Replace(Environment.NewLine, "<br />", StringComparison.Ordinal);

    private static string Encode(string value) => WebUtility.HtmlEncode(value);
}
