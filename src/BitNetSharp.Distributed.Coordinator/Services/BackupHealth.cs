using System;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Thread-safe singleton that tracks the health of the
/// <see cref="DatabaseBackupService"/> loop. Mirrors
/// <see cref="PruneHealth"/> so the dashboard can surface a streak of
/// failed backups the same way it surfaces failed prune iterations.
/// </summary>
public sealed class BackupHealth
{
    private readonly object _gate = new();
    private int _consecutiveFailures;
    private int _totalFailures;
    private string? _lastFailureMessage;
    private DateTimeOffset? _lastFailureAt;
    private DateTimeOffset? _lastSuccessAt;
    private string? _lastBackupPath;

    public void RecordSuccess(DateTimeOffset at, string backupPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(backupPath);
        lock (_gate)
        {
            _consecutiveFailures = 0;
            _lastSuccessAt = at;
            _lastBackupPath = backupPath;
        }
    }

    public void RecordFailure(Exception ex, DateTimeOffset at)
    {
        ArgumentNullException.ThrowIfNull(ex);
        lock (_gate)
        {
            _consecutiveFailures++;
            _totalFailures++;
            _lastFailureMessage = ex.Message;
            _lastFailureAt = at;
        }
    }

    public BackupHealthSnapshot Snapshot()
    {
        lock (_gate)
        {
            return new BackupHealthSnapshot(
                ConsecutiveFailures: _consecutiveFailures,
                TotalFailures:       _totalFailures,
                LastFailureMessage:  _lastFailureMessage,
                LastFailureAtUtc:    _lastFailureAt,
                LastSuccessAtUtc:    _lastSuccessAt,
                LastBackupPath:      _lastBackupPath);
        }
    }
}

/// <summary>
/// Point-in-time view of <see cref="BackupHealth"/> exposed in the
/// dashboard snapshot. When <see cref="ConsecutiveFailures"/> is
/// non-zero the dashboard surfaces a warn banner.
/// </summary>
public sealed record BackupHealthSnapshot(
    int ConsecutiveFailures,
    int TotalFailures,
    string? LastFailureMessage,
    DateTimeOffset? LastFailureAtUtc,
    DateTimeOffset? LastSuccessAtUtc,
    string? LastBackupPath);
