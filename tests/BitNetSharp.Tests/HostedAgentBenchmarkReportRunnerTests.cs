using BitNetSharp.App;
using BitNetSharp.Core;

namespace BitNetSharp.Tests;

public sealed class HostedAgentBenchmarkReportRunnerTests
{
    [Fact]
    public void ParsePerformanceRowsReadsBenchmarkDotNetMarkdownTables()
    {
        var outputDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        var resultsDirectory = Path.Combine(outputDirectory, "BenchmarkDotNet.Artifacts", "results");
        Directory.CreateDirectory(resultsDirectory);

        try
        {
            File.WriteAllText(
                Path.Combine(resultsDirectory, "BitNetSharp.App.HostedAgentResponseBenchmarks-report-github.md"),
                """
                | Method | ModelSpecifier | Mean | Error | StdDev | Allocated |
                | --- | --- | ---: | ---: | ---: | ---: |
                | **&#39;SpecFlow: Generate a response for a prompt&#39;** | **bitnet-b1.58-sharp** | **133.8 ms** | **90.20 ms** | **4.94 ms** | **37.39 MB** |
                | **&#39;SpecFlow: Generate a response for a prompt&#39;** | **traditional-local** | **144.4 ms** | **46.93 ms** | **2.57 ms** | **1.88 MB** |
                """);

            var rows = HostedAgentBenchmarkReportRunner.ParsePerformanceRows(outputDirectory);

            Assert.Collection(
                rows,
                row =>
                {
                    Assert.Equal("SpecFlow: Generate a response for a prompt", row.Operation);
                    Assert.Equal("bitnet-b1.58-sharp", row.ModelSpecifier);
                    Assert.Equal("133.8 ms", row.Mean);
                    Assert.Equal("4.94 ms", row.StdDev);
                    Assert.Equal("37.39 MB", row.Allocated);
                    Assert.Equal("BenchmarkDotNet.Artifacts/results/BitNetSharp.App.HostedAgentResponseBenchmarks-report.html", row.HtmlReportPath);
                },
                row =>
                {
                    Assert.Equal("traditional-local", row.ModelSpecifier);
                    Assert.Equal("1.88 MB", row.Allocated);
                    Assert.Equal("BenchmarkDotNet.Artifacts/results/BitNetSharp.App.HostedAgentResponseBenchmarks-report.csv", row.CsvReportPath);
                });
        }
        finally
        {
            Directory.Delete(outputDirectory, recursive: true);
        }
    }

    [Fact]
    public void WriteReportSiteCreatesHtmlMarkdownAndJsonArtifacts()
    {
        var outputDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(outputDirectory);

        try
        {
            var report = new HostedAgentBenchmarkComparisonReport(
                DateTimeOffset.Parse("2026-03-18T00:00:00Z"),
                ["hello", "how are you hosted"],
                [
                    new HostedAgentBenchmarkModelReport(
                        HostedAgentModelFactory.DefaultModelId,
                        "Paper-aligned BitNet b1.58 transformer",
                        TrainingSupported: false,
                        TrainingCompleted: false,
                        TrainingExamples: 0,
                        TrainingEpochs: 0,
                        SuccessfulQueries: 2,
                        TotalQueries: 2,
                        ExactMatches: 0,
                        AverageExpectedTokenRecall: 0.5d,
                        QueryResults:
                        [
                            new HostedAgentBenchmarkQueryResult("hello", "Hello!", "<response>", true, false, 0.5d),
                            new HostedAgentBenchmarkQueryResult("how are you hosted", "Hosted", "Hosted <details>", true, false, 0.5d)
                        ])
                ],
                [
                    new HostedAgentBenchmarkPerformanceRow(
                        "SpecFlow: Build the agent host for the selected model",
                        HostedAgentModelFactory.DefaultModelId,
                        "10.0 ms",
                        "1.0 ms",
                        "2 MB",
                        "BenchmarkDotNet.Artifacts/results/host-report.html",
                        "BenchmarkDotNet.Artifacts/results/host-report.csv",
                        "BenchmarkDotNet.Artifacts/results/host-report-github.md")
                ]);

            HostedAgentBenchmarkReportRunner.WriteReportSite(outputDirectory, report);

            var markdown = File.ReadAllText(Path.Combine(outputDirectory, "comparison-report.md"));
            var html = File.ReadAllText(Path.Combine(outputDirectory, "index.html"));
            var json = File.ReadAllText(Path.Combine(outputDirectory, "comparison-report.json"));

            Assert.Contains("BitNet benchmark comparison report", markdown, StringComparison.Ordinal);
            Assert.Contains("Expected-token recall", markdown, StringComparison.Ordinal);
            Assert.Contains("<response>", markdown, StringComparison.Ordinal);
            Assert.Contains("&lt;response&gt;", html, StringComparison.Ordinal);
            Assert.Contains("comparison-report.md", html, StringComparison.Ordinal);
            Assert.Contains("\"ModelSpecifier\": \"bitnet-b1.58-sharp\"", json, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(outputDirectory, recursive: true);
        }
    }
}
