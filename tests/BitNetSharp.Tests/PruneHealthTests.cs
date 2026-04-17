#if NET10_0_OR_GREATER
using System;
using BitNetSharp.Distributed.Coordinator.Services;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests for <see cref="PruneHealth"/>, the singleton that promotes
/// a swallowed prune-loop exception to a first-class dashboard signal.
/// </summary>
public sealed class PruneHealthTests
{
    [Fact]
    public void Fresh_instance_reports_zero_failures_and_no_timestamps()
    {
        var health = new PruneHealth();

        var snap = health.Snapshot();

        Assert.Equal(0, snap.ConsecutiveFailures);
        Assert.Equal(0, snap.TotalFailures);
        Assert.Null(snap.LastFailureMessage);
        Assert.Null(snap.LastFailureAtUtc);
        Assert.Null(snap.LastSuccessAtUtc);
    }

    [Fact]
    public void RecordFailure_increments_counters_and_remembers_last_message()
    {
        var health = new PruneHealth();
        var at = DateTimeOffset.UtcNow;

        health.RecordFailure(new InvalidOperationException("disk full"), at);
        health.RecordFailure(new InvalidOperationException("still full"), at.AddMinutes(1));

        var snap = health.Snapshot();
        Assert.Equal(2, snap.ConsecutiveFailures);
        Assert.Equal(2, snap.TotalFailures);
        Assert.Equal("still full", snap.LastFailureMessage);
        Assert.Equal(at.AddMinutes(1), snap.LastFailureAtUtc);
    }

    [Fact]
    public void RecordSuccess_resets_consecutive_streak_but_preserves_total()
    {
        var health = new PruneHealth();
        var start = DateTimeOffset.UtcNow;

        health.RecordFailure(new InvalidOperationException("boom"), start);
        health.RecordFailure(new InvalidOperationException("boom again"), start.AddMinutes(1));
        health.RecordSuccess(start.AddMinutes(2));

        var snap = health.Snapshot();
        Assert.Equal(0, snap.ConsecutiveFailures);
        Assert.Equal(2, snap.TotalFailures);
        Assert.Equal(start.AddMinutes(2), snap.LastSuccessAtUtc);
        // Last-failure fields remain so operators can see the prior
        // incident even after recovery.
        Assert.Equal("boom again", snap.LastFailureMessage);
    }

    [Fact]
    public void RecordFailure_after_recovery_starts_a_new_streak()
    {
        var health = new PruneHealth();
        var t = DateTimeOffset.UtcNow;

        health.RecordFailure(new InvalidOperationException("first"), t);
        health.RecordSuccess(t.AddMinutes(1));
        health.RecordFailure(new InvalidOperationException("second"), t.AddMinutes(2));

        var snap = health.Snapshot();
        Assert.Equal(1, snap.ConsecutiveFailures);
        Assert.Equal(2, snap.TotalFailures);
        Assert.Equal("second", snap.LastFailureMessage);
    }
}
#endif
