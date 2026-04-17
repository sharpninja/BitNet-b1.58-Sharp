using System;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Thread-safe singleton that tracks the health of the
/// <see cref="TelemetryPruneService"/> loop. A swallowed exception in
/// the prune body used to vanish into a warn-level log line; this
/// service promotes it to a first-class counter + last-message the
/// dashboard can show. A streak of consecutive failures means the
/// SQLite file is growing unbounded and an operator needs to look.
/// </summary>
public sealed class PruneHealth
{
    private readonly object _gate = new();
    private int _consecutiveFailures;
    private int _totalFailures;
    private string? _lastFailureMessage;
    private DateTimeOffset? _lastFailureAt;
    private DateTimeOffset? _lastSuccessAt;

    public void RecordSuccess(DateTimeOffset at)
    {
        lock (_gate)
        {
            _consecutiveFailures = 0;
            _lastSuccessAt = at;
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

    public PruneHealthSnapshot Snapshot()
    {
        lock (_gate)
        {
            return new PruneHealthSnapshot(
                ConsecutiveFailures: _consecutiveFailures,
                TotalFailures:       _totalFailures,
                LastFailureMessage:  _lastFailureMessage,
                LastFailureAtUtc:    _lastFailureAt,
                LastSuccessAtUtc:    _lastSuccessAt);
        }
    }
}

/// <summary>
/// Point-in-time view of <see cref="PruneHealth"/> exposed in the
/// dashboard snapshot. When <see cref="ConsecutiveFailures"/> is
/// non-zero the dashboard surfaces a warn banner with the last error.
/// </summary>
public sealed record PruneHealthSnapshot(
    int ConsecutiveFailures,
    int TotalFailures,
    string? LastFailureMessage,
    DateTimeOffset? LastFailureAtUtc,
    DateTimeOffset? LastSuccessAtUtc);
