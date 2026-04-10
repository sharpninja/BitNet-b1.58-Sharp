using System.Net;
using System.Text;
using System.Text.Json;
using System.Globalization;
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
    int BenchmarkPromptTokenCount = 0,
    double? EstimatedResidentModelMegabytes = null,
    double? ChainBucketAcceptanceRate = null)
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
    double? EstimatedResidentModelMegabytes,
    double? WikiText2Perplexity,
    double? ChainBucketAcceptanceRate = null);

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
    HostedAgentBenchmarkComparisonSummary? ComparisonSummary = null,
    string TrainingDataset = "BitNetTrainingCorpus.CreateDefaultExamples()");

public sealed record HostedAgentBenchmarkComparisonSummary(
    string PerplexityDataset,
    IReadOnlyList<HostedAgentBenchmarkComparisonMetric> Models,
    double? BitNetSpeedupVersusTraditional,
    double? BitNetMemoryDeltaPercentVersusTraditional,
    double? BitNetResidentModelMemoryIncreasePercentVersusTraditional,
    double? BitNetQualityDeltaPercentVersusTraditional);

public static class HostedAgentBenchmarkReportRunner
{
    private const string StampedJsonTimestampFormat = "yyyyMMddTHHmmssZ";
    private const string ResponseOperation = "SpecFlow: Generate a response for a prompt";
    private const string TrainingOperation = "SpecFlow: Train the selected model on the TinyLlama-1.1B benchmark dataset";
    private const double NanosecondsPerMillisecond = 1_000_000d;
    private const double MicrosecondsPerMillisecond = 1_000d;
    private const double MillisecondsPerSecond = 1_000d;
    private const double BytesPerMegabyte = 1024d * 1024d;
    private const double KilobytesPerMegabyte = 1024d;
    public const double DefaultPerplexitySamplePercent = 10d;

    public static async Task<string> RunAsync(
        HostedAgentBenchmarkOptions options,
        string? outputDirectory,
        string? commitHash = null,
        double perplexitySamplePercent = DefaultPerplexitySamplePercent,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (perplexitySamplePercent <= 0d || perplexitySamplePercent > 100d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(perplexitySamplePercent),
                perplexitySamplePercent,
                "perplexitySamplePercent must be in the range (0, 100].");
        }

        var originalWorkingDirectory = Directory.GetCurrentDirectory();
        var reportDirectory = Path.GetFullPath(
            string.IsNullOrWhiteSpace(outputDirectory)
                ? Path.Combine(originalWorkingDirectory, "artifacts", "benchmark-report")
                : outputDirectory);
        Directory.CreateDirectory(reportDirectory);

        var progressLogPath = Path.Combine(reportDirectory, "progress.log");
        LogProgress(progressLogPath, $"RunAsync started (perplexitySamplePercent={perplexitySamplePercent:0.##}%)");

        LogProgress(progressLogPath, "Phase 1: BenchmarkDotNet suite starting");
        HostedAgentBenchmarkRunner.Run(options);
        LogProgress(progressLogPath, "Phase 1: BenchmarkDotNet suite complete");

        LogProgress(progressLogPath, "Phase 2: Copying BDN artifacts");
        CopyArtifactsDirectory(
            Path.Combine(originalWorkingDirectory, "BenchmarkDotNet.Artifacts"),
            Path.Combine(reportDirectory, "BenchmarkDotNet.Artifacts"));
        LogProgress(progressLogPath, "Phase 2: BDN artifacts copied");

        LogProgress(progressLogPath, "Phase 3: Creating benchmark examples");
        var trainingExamples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        LogProgress(progressLogPath, $"Phase 3: Created {trainingExamples.Count} training examples");

        LogProgress(progressLogPath, "Phase 4: CreateModelReportsAsync starting");
        var modelReports = await CreateModelReportsAsync(options, trainingExamples, progressLogPath, perplexitySamplePercent, cancellationToken);
        LogProgress(progressLogPath, $"Phase 4: CreateModelReportsAsync complete ({modelReports.Count} reports)");

        // Save intermediate model reports in case the rest fails.
        SaveIntermediate(reportDirectory, "model-reports.json", modelReports);

        LogProgress(progressLogPath, "Phase 5: Parsing performance rows");
        var performanceRows = ParsePerformanceRows(reportDirectory);
        LogProgress(progressLogPath, $"Phase 5: Parsed {performanceRows.Count} performance rows");
        SaveIntermediate(reportDirectory, "performance-rows.json", performanceRows);

        LogProgress(progressLogPath, "Phase 6: Creating comparison summary");
        var comparisonSummary = CreateComparisonSummary(modelReports, performanceRows);
        LogProgress(progressLogPath, "Phase 6: Comparison summary complete");

        var report = new HostedAgentBenchmarkComparisonReport(
            DateTimeOffset.UtcNow,
            trainingExamples.Select(static example => example.Prompt).ToArray(),
            modelReports,
            performanceRows,
            comparisonSummary,
            BitNetTrainingCorpus.BenchmarkDatasetName);

        LogProgress(progressLogPath, "Phase 7: Writing report site");
        WriteReportSite(reportDirectory, report, commitHash);
        LogProgress(progressLogPath, "Phase 7: Report site written");

        LogProgress(progressLogPath, "RunAsync complete");
        return reportDirectory;
    }

    private static void LogProgress(string path, string message)
    {
        var timestamp = DateTimeOffset.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ");
        var line = $"{timestamp}  {message}";
        try
        {
            File.AppendAllText(path, line + Environment.NewLine);
        }
        catch
        {
            // Ignore log failures.
        }

        Console.WriteLine($"[PROGRESS] {line}");
    }

    private static void SaveIntermediate<T>(string reportDirectory, string fileName, T data)
    {
        try
        {
            var path = Path.Combine(reportDirectory, fileName);
            File.WriteAllText(path, JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[PROGRESS] Failed to save intermediate {fileName}: {ex.Message}");
        }
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
        string progressLogPath,
        double perplexitySamplePercent,
        CancellationToken cancellationToken)
    {
        var reports = new List<HostedAgentBenchmarkModelReport>();
        var reportDirectory = Path.GetDirectoryName(progressLogPath) ?? string.Empty;

        foreach (var modelSpecifier in options.ModelSpecifiers)
        {
            LogProgress(progressLogPath, $"Model '{modelSpecifier}': creating prepared model");
            using var model = HostedAgentBenchmarkModelBootstrap.CreatePreparedModel(modelSpecifier, options, trainingExamples);
            LogProgress(progressLogPath, $"Model '{modelSpecifier}': prepared model created");

            var trainingSupported = model is ITrainableHostedAgentModel;
            var trainingCompleted = false;
            var trainingEpochs = 0;
            if (options.EnableBucketing && model is BitNetHostedAgentModel preTrainedBitNetModel)
            {
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': mining pre-training buckets");
                preTrainedBitNetModel.Model.MineAndLoadBuckets(trainingExamples);
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': pre-training buckets mined");
            }

            if (trainingSupported)
            {
                trainingEpochs = GetTrainingEpochs(model);
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': training starting ({trainingEpochs} epochs)");
                ((ITrainableHostedAgentModel)model).Train(trainingExamples, trainingEpochs);
                trainingCompleted = true;
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': training complete");

                if (options.EnableBucketing && model is BitNetHostedAgentModel trainedBitNetModel)
                {
                    LogProgress(progressLogPath, $"Model '{modelSpecifier}': mining post-training buckets");
                    trainedBitNetModel.Model.MineAndLoadBuckets(trainingExamples);
                    LogProgress(progressLogPath, $"Model '{modelSpecifier}': post-training buckets mined");
                }
            }

            LogProgress(progressLogPath, $"Model '{modelSpecifier}': running query examples");
            var queryResults = new List<HostedAgentBenchmarkQueryResult>(trainingExamples.Count);
            var attemptedChainTokens = 0;
            var acceptedChainTokens = 0;
            var queryIndex = 0;
            foreach (var example in trainingExamples)
            {
                queryIndex++;
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': query {queryIndex}/{trainingExamples.Count}");
                string responseText;
                if (model is BitNetHostedAgentModel bitNetModel)
                {
                    var generationResult = bitNetModel.Model.GenerateResponse(example.Prompt, options.MaxOutputTokens);
                    responseText = generationResult.ResponseText;
                    if (generationResult.ChainBucketMetrics is not null)
                    {
                        attemptedChainTokens += generationResult.ChainBucketMetrics.AttemptedTokens;
                        acceptedChainTokens += generationResult.ChainBucketMetrics.AcceptedTokens;
                    }
                }
                else
                {
                    var response = await model.GetResponseAsync(example.Prompt, options.MaxOutputTokens, cancellationToken);
                    responseText = response.Text;
                }

                var exactMatch = Normalize(responseText) == Normalize(example.Response);
                queryResults.Add(new HostedAgentBenchmarkQueryResult(
                    example.Prompt,
                    example.Response,
                    responseText,
                    !string.IsNullOrWhiteSpace(responseText),
                    exactMatch,
                    CalculateExpectedTokenRecall(responseText, example.Response)));
            }

            LogProgress(progressLogPath, $"Model '{modelSpecifier}': queries complete");

            double? chainBucketAcceptanceRate = null;
            if (attemptedChainTokens > 0)
            {
                chainBucketAcceptanceRate = acceptedChainTokens / (double)attemptedChainTokens;
            }

            LogProgress(progressLogPath, $"Model '{modelSpecifier}': computing perplexity ({perplexitySamplePercent:0.##}% of WikiText2 validation)");
            var perplexity = GetWikiText2Perplexity(model, perplexitySamplePercent);
            LogProgress(progressLogPath, $"Model '{modelSpecifier}': perplexity = {perplexity}");

            LogProgress(progressLogPath, $"Model '{modelSpecifier}': computing benchmark prompt token count");
            var promptTokenCount = await GetBenchmarkPromptTokenCountAsync(model, options, cancellationToken);
            LogProgress(progressLogPath, $"Model '{modelSpecifier}': prompt token count = {promptTokenCount}");

            LogProgress(progressLogPath, $"Model '{modelSpecifier}': estimating resident model megabytes");
            var residentMb = GetEstimatedResidentModelMegabytes(model);

            BitNetPaperAuditReport? auditReport = null;
            if (model is BitNetHostedAgentModel auditBitNetModel)
            {
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': running paper alignment audit");
                auditReport = BitNetPaperAuditor.CreateReport(auditBitNetModel.Model);
                LogProgress(progressLogPath, $"Model '{modelSpecifier}': audit complete");
            }

            var modelReport = new HostedAgentBenchmarkModelReport(
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
                auditReport,
                perplexity,
                promptTokenCount,
                residentMb,
                chainBucketAcceptanceRate);

            reports.Add(modelReport);
            LogProgress(progressLogPath, $"Model '{modelSpecifier}': report added ({reports.Count}/{options.ModelSpecifiers.Count})");

            // Persist incremental reports in case a subsequent model crashes.
            if (!string.IsNullOrEmpty(reportDirectory))
            {
                SaveIntermediate(reportDirectory, $"model-report-{reports.Count:D2}-{modelSpecifier.Replace('/', '_').Replace('.', '_')}.json", modelReport);
            }
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
                    modelReport.EstimatedResidentModelMegabytes,
                    modelReport.WikiText2Perplexity,
                    modelReport.ChainBucketAcceptanceRate);
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
            bitNetMetric?.EstimatedResidentModelMegabytes is > 0d && traditionalMetric?.EstimatedResidentModelMegabytes is > 0d
                ? ((bitNetMetric.EstimatedResidentModelMegabytes.Value - traditionalMetric.EstimatedResidentModelMegabytes.Value)
                    / traditionalMetric.EstimatedResidentModelMegabytes.Value) * 100d
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
        builder.AppendLine($"- Training set: `{report.TrainingDataset}`");
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
        builder.AppendLine($"    <li>Training set: <code>{Encode(report.TrainingDataset)}</code></li>");
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

    // Use the configured percentage of the WikiText2 validation set for the benchmark-report
    // perplexity calculation, stride-sampled evenly across the full validation set so coverage
    // is representative of the entire corpus. The full 3,760-sample set takes hours to evaluate
    // on a consumer CPU; 10% (376 entries) runs in a few minutes and is sufficient for relative
    // comparison between models.
    private static IReadOnlyList<string> GetBenchmarkWikiText2ValidationSamples(double samplePercent)
    {
        var all = BitNetBenchmarkFixtures.WikiText2ValidationSamples;
        var targetCount = Math.Max(1, (int)Math.Ceiling(all.Count * (samplePercent / 100d)));
        var stride = Math.Max(1, all.Count / targetCount);
        var samples = new List<string>(targetCount);
        for (var i = 0; i < all.Count && samples.Count < targetCount; i += stride)
        {
            samples.Add(all[i]);
        }

        return samples;
    }

    private static double? GetWikiText2Perplexity(IHostedAgentModel model, double samplePercent)
    {
        var samples = GetBenchmarkWikiText2ValidationSamples(samplePercent);
        return model switch
        {
            BitNetHostedAgentModel bitNetModel => bitNetModel.Model.CalculatePerplexity(samples),
            TraditionalLocalHostedAgentModel traditionalModel => traditionalModel.Model.CalculatePerplexity(samples),
            _ => null
        };
    }

    private static double? GetEstimatedResidentModelMegabytes(IHostedAgentModel model)
    {
        var bytes = model switch
        {
            BitNetHostedAgentModel bitNetModel => bitNetModel.Model.EstimateResidentParameterBytes(),
            TraditionalLocalHostedAgentModel traditionalModel => traditionalModel.Model.EstimateResidentParameterBytes(),
            _ => 0L
        };

        return bytes > 0 ? bytes / BytesPerMegabyte : null;
    }

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
        builder.AppendLine("| Model | Response mean | Response tokens/sec | Training mean | Perplexity | Response allocated | Estimated resident model memory |");
        builder.AppendLine("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
        foreach (var model in summary.Models)
        {
            builder.AppendLine(
                $"| {model.ModelSpecifier} | {model.ResponseMean ?? "-"} | {FormatNullableNumber(model.ResponseTokensPerSecond)} | {model.TrainingMean ?? "-"} | {FormatNullableNumber(model.WikiText2Perplexity)} | {FormatNullableMegabytes(model.ResponseAllocatedMegabytes)} | {FormatNullableMegabytes(model.EstimatedResidentModelMegabytes)} |");
        }

        builder.AppendLine();
        builder.AppendLine("| Delta | Value |");
        builder.AppendLine("| --- | ---: |");
        builder.AppendLine($"| BitNet speedup vs traditional | {FormatNullableRatio(summary.BitNetSpeedupVersusTraditional)} |");
        builder.AppendLine($"| BitNet memory reduction vs traditional | {FormatNullablePercent(summary.BitNetMemoryDeltaPercentVersusTraditional)} |");
        builder.AppendLine($"| BitNet resident model memory increase vs traditional | {FormatNullablePercent(summary.BitNetResidentModelMemoryIncreasePercentVersusTraditional)} |");
        builder.AppendLine($"| BitNet quality improvement vs traditional | {FormatNullablePercent(summary.BitNetQualityDeltaPercentVersusTraditional)} |");

        if (summary.Models.Any(static model => model.ChainBucketAcceptanceRate is not null))
        {
            builder.AppendLine();
            builder.AppendLine("### Chain-bucket speculation");
            builder.AppendLine();
            builder.AppendLine("| Model | Chain acceptance rate | Response tokens/sec |");
            builder.AppendLine("| --- | ---: | ---: |");
            foreach (var model in summary.Models)
            {
                builder.AppendLine(
                    $"| {model.ModelSpecifier} | {FormatNullableRate(model.ChainBucketAcceptanceRate)} | {FormatNullableNumber(model.ResponseTokensPerSecond)} |");
            }
        }
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
        builder.AppendLine("    <thead><tr><th>Model</th><th>Response mean</th><th>Response tokens/sec</th><th>Training mean</th><th>Perplexity</th><th>Response allocated</th><th>Estimated resident model memory</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        foreach (var model in summary.Models)
        {
            builder.AppendLine(
                $"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{Encode(model.ResponseMean ?? "-")}</td><td>{Encode(FormatNullableNumber(model.ResponseTokensPerSecond))}</td><td>{Encode(model.TrainingMean ?? "-")}</td><td>{Encode(FormatNullableNumber(model.WikiText2Perplexity))}</td><td>{Encode(FormatNullableMegabytes(model.ResponseAllocatedMegabytes))}</td><td>{Encode(FormatNullableMegabytes(model.EstimatedResidentModelMegabytes))}</td></tr>");
        }

        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");
        builder.AppendLine("  <table>");
        builder.AppendLine("    <thead><tr><th>Delta</th><th>Value</th></tr></thead>");
        builder.AppendLine("    <tbody>");
        builder.AppendLine($"      <tr><td>BitNet speedup vs traditional</td><td>{Encode(FormatNullableRatio(summary.BitNetSpeedupVersusTraditional))}</td></tr>");
        builder.AppendLine($"      <tr><td>BitNet memory reduction vs traditional</td><td>{Encode(FormatNullablePercent(summary.BitNetMemoryDeltaPercentVersusTraditional))}</td></tr>");
        builder.AppendLine($"      <tr><td>BitNet resident model memory increase vs traditional</td><td>{Encode(FormatNullablePercent(summary.BitNetResidentModelMemoryIncreasePercentVersusTraditional))}</td></tr>");
        builder.AppendLine($"      <tr><td>BitNet quality improvement vs traditional</td><td>{Encode(FormatNullablePercent(summary.BitNetQualityDeltaPercentVersusTraditional))}</td></tr>");
        builder.AppendLine("    </tbody>");
        builder.AppendLine("  </table>");

        if (summary.Models.Any(static model => model.ChainBucketAcceptanceRate is not null))
        {
            builder.AppendLine("  <h3>Chain-bucket speculation</h3>");
            builder.AppendLine("  <table>");
            builder.AppendLine("    <thead><tr><th>Model</th><th>Chain acceptance rate</th><th>Response tokens/sec</th></tr></thead>");
            builder.AppendLine("    <tbody>");
            foreach (var model in summary.Models)
            {
                builder.AppendLine(
                    $"      <tr><td>{Encode(model.ModelSpecifier)}</td><td>{Encode(FormatNullableRate(model.ChainBucketAcceptanceRate))}</td><td>{Encode(FormatNullableNumber(model.ResponseTokensPerSecond))}</td></tr>");
            }

            builder.AppendLine("    </tbody>");
            builder.AppendLine("  </table>");
        }

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
        AppendBarChart(builder, "Estimated resident model memory (MB)", summary.Models, summary.Models.Max(static model => model.EstimatedResidentModelMegabytes ?? 0d), 0d, static model => model.EstimatedResidentModelMegabytes, FormatNullableMegabytes);
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

    private static string FormatNullableNumber(double? value) =>
        value is null ? "-" : value.Value.ToString("0.##", CultureInfo.InvariantCulture);

    private static string FormatNullableMegabytes(double? value) =>
        value is null ? "-" : $"{value.Value.ToString("0.##", CultureInfo.InvariantCulture)} MB";

    private static string FormatNullablePercent(double? value) =>
        value is null ? "-" : $"{value.Value.ToString("0.##", CultureInfo.InvariantCulture)}%";

    private static string FormatNullableRate(double? value) =>
        value is null ? "-" : $"{(value.Value * 100d).ToString("0.0", CultureInfo.InvariantCulture)}%";

    private static string FormatNullableRatio(double? value) =>
        value is null ? "-" : $"{value.Value.ToString("0.##", CultureInfo.InvariantCulture)}x";

    private static string Encode(string value) => WebUtility.HtmlEncode(value);
}
