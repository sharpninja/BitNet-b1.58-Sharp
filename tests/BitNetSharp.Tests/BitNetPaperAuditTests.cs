using BitNetSharp.App;
using BitNetSharp.Core;

namespace BitNetSharp.Tests;

public sealed class BitNetPaperAuditTests
{
    private const string BlankSeparatorLine = " ";

    private static IReadOnlyList<BitNetBenchmarkTextFixture> CreateCompactPerplexityDatasets() =>
    [
        new(
            "WikiText2",
            BitNetBenchmarkFixtures.WikiText2ValidationSamples
                .Where(static sample => string.Equals(sample, BlankSeparatorLine, StringComparison.Ordinal)
                    || sample.StartsWith(" = ", StringComparison.Ordinal))
                .Take(8)
                .ToArray()),
        new("C4", BitNetBenchmarkFixtures.C4ValidationSamples),
        new("RedPajama", BitNetBenchmarkFixtures.RedPajamaValidationSamples)
    ];

    [Fact]
    public void PaperAuditPassesArchitectureChecksAndReportsRuntimeCoverage()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);

        var report = BitNetPaperAuditor.CreateReport(model, perplexityDatasets: CreateCompactPerplexityDatasets());

        Assert.True(report.ArchitectureChecksPassed);
        Assert.Equal(0, report.Checks.Count(c => !string.Equals(c.Area, "Memory", StringComparison.Ordinal) && c.Status == BitNetPaperAuditStatus.Failed));
        Assert.True(report.PassedCount >= 10);
        Assert.Equal(0, report.PendingCount);
        Assert.Contains(
            report.Checks,
            check => check.Status == BitNetPaperAuditStatus.Passed
                && check.Requirement.Contains("Zero-shot benchmark fixtures", StringComparison.Ordinal));
    }

    [Fact]
    public void PaperAuditExplainsResidentMemoryDeltaVersusTraditionalModel()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);

        var report = BitNetPaperAuditor.CreateReport(model, perplexityDatasets: CreateCompactPerplexityDatasets());

        Assert.Contains(
            report.Checks,
            check => check.Area == "Memory"
                && check.Details.Contains("traditional-local", StringComparison.Ordinal)
                && check.Details.Contains("BitLinear projections", StringComparison.Ordinal));
    }

    [Fact]
    public void PaperAuditCommandFormatterIncludesStatusSummary()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);
        var report = BitNetPaperAuditor.CreateReport(model, perplexityDatasets: CreateCompactPerplexityDatasets());

        var formatted = BitNetPaperAuditCommand.FormatReport(report);

        Assert.Contains("Paper-alignment audit: bitnet-b1.58-sharp", formatted, StringComparison.Ordinal);
        Assert.Contains("Passed:", formatted, StringComparison.Ordinal);
        Assert.Contains("Pending:", formatted, StringComparison.Ordinal);
        Assert.Contains("[PASS] Architecture", formatted, StringComparison.Ordinal);
        Assert.DoesNotContain("[PENDING]", formatted, StringComparison.Ordinal);
        Assert.Contains("[PASS] Benchmark pipeline", formatted, StringComparison.Ordinal);
    }
}
