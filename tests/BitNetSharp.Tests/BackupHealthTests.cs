using System;
using BitNetSharp.Distributed.Coordinator.Services;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests for <see cref="BackupHealth"/>, the dashboard signal
/// mirroring <see cref="PruneHealth"/> but for the nightly backup loop.
/// </summary>
public sealed class BackupHealthTests
{
    [Fact]
    public void Fresh_instance_reports_zero_failures_and_no_timestamps()
    {
        var health = new BackupHealth();

        var snap = health.Snapshot();

        Assert.Equal(0, snap.ConsecutiveFailures);
        Assert.Equal(0, snap.TotalFailures);
        Assert.Null(snap.LastFailureMessage);
        Assert.Null(snap.LastFailureAtUtc);
        Assert.Null(snap.LastSuccessAtUtc);
        Assert.Null(snap.LastBackupPath);
    }

    [Fact]
    public void RecordFailure_increments_counters_and_remembers_last_message()
    {
        var health = new BackupHealth();
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
    public void RecordSuccess_resets_consecutive_streak_and_records_path()
    {
        var health = new BackupHealth();
        var start = DateTimeOffset.UtcNow;

        health.RecordFailure(new InvalidOperationException("boom"), start);
        health.RecordSuccess(start.AddMinutes(1), @"C:\backups\20260417T020000");

        var snap = health.Snapshot();
        Assert.Equal(0, snap.ConsecutiveFailures);
        Assert.Equal(1, snap.TotalFailures);
        Assert.Equal(start.AddMinutes(1), snap.LastSuccessAtUtc);
        Assert.Equal(@"C:\backups\20260417T020000", snap.LastBackupPath);
        Assert.Equal("boom", snap.LastFailureMessage);
    }

    [Fact]
    public void RecordFailure_after_recovery_starts_a_new_streak()
    {
        var health = new BackupHealth();
        var t = DateTimeOffset.UtcNow;

        health.RecordFailure(new InvalidOperationException("first"), t);
        health.RecordSuccess(t.AddMinutes(1), "/var/backups/x");
        health.RecordFailure(new InvalidOperationException("second"), t.AddMinutes(2));

        var snap = health.Snapshot();
        Assert.Equal(1, snap.ConsecutiveFailures);
        Assert.Equal(2, snap.TotalFailures);
        Assert.Equal("second", snap.LastFailureMessage);
        Assert.Equal("/var/backups/x", snap.LastBackupPath);
    }
}
