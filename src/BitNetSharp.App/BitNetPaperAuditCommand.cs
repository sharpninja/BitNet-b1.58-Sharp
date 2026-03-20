using System.Text;
using BitNetSharp.Core;

namespace BitNetSharp.App;

public static class BitNetPaperAuditCommand
{
    public static string FormatReport(BitNetPaperAuditReport report)
    {
        ArgumentNullException.ThrowIfNull(report);

        var builder = new StringBuilder();
        builder.AppendLine($"Paper-alignment audit: {report.ModelId}");
        builder.AppendLine(report.DisplayName);
        builder.AppendLine($"Passed: {report.PassedCount}");
        builder.AppendLine($"Pending: {report.PendingCount}");
        builder.AppendLine($"Failed: {report.FailedCount}");
        builder.AppendLine();

        foreach (var check in report.Checks)
        {
            builder.AppendLine($"[{FormatStatus(check.Status)}] {check.Area} - {check.Requirement}");
            builder.AppendLine($"  {check.Details}");
        }

        return builder.ToString().TrimEnd();
    }

    private static string FormatStatus(BitNetPaperAuditStatus status) => status switch
    {
        BitNetPaperAuditStatus.Passed => "PASS",
        BitNetPaperAuditStatus.Pending => "PENDING",
        BitNetPaperAuditStatus.Failed => "FAIL",
        _ => status.ToString().ToUpperInvariant()
    };
}
