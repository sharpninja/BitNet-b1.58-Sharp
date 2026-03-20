using BitNetSharp.App;
using BitNetSharp.Core;

namespace BitNetSharp.Tests;

public sealed class BitNetPaperAuditTests
{
    [Fact]
    public void PaperAuditPassesArchitectureChecksAndReportsRuntimeCoverage()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);

        var report = BitNetPaperAuditor.CreateReport(model);

        Assert.True(report.ArchitectureChecksPassed);
        Assert.Equal(0, report.FailedCount);
        Assert.True(report.PassedCount >= 10);
        Assert.Equal(0, report.PendingCount);
        Assert.Contains(
            report.Checks,
            check => check.Status == BitNetPaperAuditStatus.Passed
                && check.Requirement.Contains("Zero-shot benchmark fixtures", StringComparison.Ordinal));
    }

    [Fact]
    public void PaperAuditCommandFormatterIncludesStatusSummary()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);
        var report = BitNetPaperAuditor.CreateReport(model);

        var formatted = BitNetPaperAuditCommand.FormatReport(report);

        Assert.Contains("Paper-alignment audit: bitnet-b1.58-sharp", formatted, StringComparison.Ordinal);
        Assert.Contains("Passed:", formatted, StringComparison.Ordinal);
        Assert.Contains("Pending:", formatted, StringComparison.Ordinal);
        Assert.Contains("[PASS] Architecture", formatted, StringComparison.Ordinal);
        Assert.DoesNotContain("[PENDING]", formatted, StringComparison.Ordinal);
        Assert.Contains("[PASS] Benchmark pipeline", formatted, StringComparison.Ordinal);
    }
}
