#if NET10_0_OR_GREATER
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests exercising the SQLite work queue store end to end.
/// Each test gets its own temp file so the WAL-mode DB does not leak
/// state between tests. The <see cref="FakeTimeProvider"/> lets the
/// deadline-expiry test advance the clock deterministically without
/// sleeping.
/// </summary>
public sealed class SqliteWorkQueueStoreTests : IDisposable
{
    private readonly string _databasePath;
    private readonly string _connectionString;
    private readonly FakeTimeProvider _time;
    private readonly SqliteWorkQueueStore _store;

    public SqliteWorkQueueStoreTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-wq-{Guid.NewGuid():N}.db");
        _connectionString = $"Data Source={_databasePath}";
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 17, 0, 0, TimeSpan.Zero));
        _store = new SqliteWorkQueueStore(_connectionString, _time);
    }

    public void Dispose()
    {
        _store.Dispose();
        TryDelete(_databasePath);
        TryDelete(_databasePath + "-wal");
        TryDelete(_databasePath + "-shm");
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
    }

    private WorkerRecord NewWorker(string workerId)
    {
        var now = _time.GetUtcNow();
        return new WorkerRecord(
            WorkerId: workerId,
            Name: workerId,
            CpuThreads: 4,
            TokensPerSecond: 1000.0,
            RecommendedTokensPerTask: 4096,
            ProcessArchitecture: "X64",
            OsDescription: "test",
            RegisteredAtUtc: now,
            LastHeartbeatUtc: now,
            State: WorkerState.Active);
    }

    private WorkTaskRecord NewPendingTask(string taskId, long weightVersion = 1)
    {
        return new WorkTaskRecord(
            TaskId: taskId,
            WeightVersion: weightVersion,
            ShardId: "shard-A",
            ShardOffset: 0,
            ShardLength: 1024,
            TokensPerTask: 4096,
            KLocalSteps: 4,
            HyperparametersJson: "{\"lr\":1e-3}",
            State: WorkTaskState.Pending,
            AssignedWorkerId: null,
            AssignedAtUtc: null,
            DeadlineUtc: null,
            Attempt: 0,
            CreatedAtUtc: _time.GetUtcNow(),
            CompletedAtUtc: null);
    }

    // ── enqueue / count ─────────────────────────────────────────────

    [Fact]
    public void EnqueuePending_inserts_row_visible_in_count()
    {
        _store.EnqueuePending(NewPendingTask("task-001"));

        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending));
        Assert.Equal(0, _store.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void CountByTaskIdPrefixAndState_matches_prefix_only()
    {
        _store.EnqueuePending(NewPendingTask("task-seed-a"));
        _store.EnqueuePending(NewPendingTask("task-seed-b"));
        _store.EnqueuePending(NewPendingTask("task-real-1"));
        // Drive seed-a and real-1 to Done.
        var c1 = _store.TryClaimNextPending("w", TimeSpan.FromMinutes(10));
        _store.MarkCompleted(c1!.TaskId, "w");
        var c2 = _store.TryClaimNextPending("w", TimeSpan.FromMinutes(10));
        _store.MarkCompleted(c2!.TaskId, "w");
        var c3 = _store.TryClaimNextPending("w", TimeSpan.FromMinutes(10));
        _store.MarkCompleted(c3!.TaskId, "w");

        Assert.Equal(2, _store.CountByTaskIdPrefixAndState("task-seed-", WorkTaskState.Done));
        Assert.Equal(0, _store.CountByTaskIdPrefixAndState("task-seed-", WorkTaskState.Pending));
    }

    [Fact]
    public void MarkLegacyByTaskIdPrefix_excludes_rows_from_excludeLegacy_count()
    {
        _store.EnqueuePending(NewPendingTask("task-seed-a"));
        _store.EnqueuePending(NewPendingTask("task-real-1"));

        var marked = _store.MarkLegacyByTaskIdPrefix("task-seed-");

        Assert.Equal(1, marked);
        Assert.Equal(2, _store.CountByState(WorkTaskState.Pending));
        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending, excludeLegacy: true));
    }

    [Fact]
    public void MarkLegacyByTaskIdPrefix_is_idempotent()
    {
        _store.EnqueuePending(NewPendingTask("task-seed-a"));
        Assert.Equal(1, _store.MarkLegacyByTaskIdPrefix("task-seed-"));
        // Re-running marks nothing — the WHERE legacy=0 guard skips already-tagged rows.
        Assert.Equal(0, _store.MarkLegacyByTaskIdPrefix("task-seed-"));
    }

    [Fact]
    public void UnmarkLegacyByTaskIdPrefix_restores_row_to_counter()
    {
        _store.EnqueuePending(NewPendingTask("task-seed-a"));
        _store.MarkLegacyByTaskIdPrefix("task-seed-");
        Assert.Equal(0, _store.CountByState(WorkTaskState.Pending, excludeLegacy: true));

        Assert.Equal(1, _store.UnmarkLegacyByTaskIdPrefix("task-seed-"));
        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending, excludeLegacy: true));
    }

    [Fact]
    public void DeleteByTaskIdPrefixAndState_removes_matching_rows_only()
    {
        _store.EnqueuePending(NewPendingTask("task-seed-a"));
        _store.EnqueuePending(NewPendingTask("task-real-1"));
        var c1 = _store.TryClaimNextPending("w", TimeSpan.FromMinutes(10));
        _store.MarkCompleted(c1!.TaskId, "w");
        var c2 = _store.TryClaimNextPending("w", TimeSpan.FromMinutes(10));
        _store.MarkCompleted(c2!.TaskId, "w");

        var deleted = _store.DeleteByTaskIdPrefixAndState("task-seed-", WorkTaskState.Done);

        Assert.Equal(1, deleted);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Done));
        Assert.NotNull(_store.GetById("task-real-1"));
        Assert.Null(_store.GetById("task-seed-a"));
    }

    [Fact]
    public void EnqueuePending_rejects_non_pending_task()
    {
        var brokenTask = NewPendingTask("task-002") with { State = WorkTaskState.Done };
        Assert.Throws<ArgumentException>(() => _store.EnqueuePending(brokenTask));
    }

    // ── claim (dequeue) ─────────────────────────────────────────────

    [Fact]
    public void TryClaimNextPending_returns_null_when_queue_is_empty()
    {
        var claimed = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));
        Assert.Null(claimed);
    }

    [Fact]
    public void TryClaimNextPending_transitions_pending_to_assigned()
    {
        _store.EnqueuePending(NewPendingTask("task-003"));

        var claimed = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));

        Assert.NotNull(claimed);
        Assert.Equal("task-003", claimed!.TaskId);
        Assert.Equal(WorkTaskState.Assigned, claimed.State);
        Assert.Equal("worker-1", claimed.AssignedWorkerId);
        Assert.NotNull(claimed.DeadlineUtc);
        Assert.Equal(1, claimed.Attempt);
        Assert.Equal(0, _store.CountByState(WorkTaskState.Pending));
        Assert.Equal(1, _store.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void TryClaimNextPending_respects_fifo_by_created_at()
    {
        var earliest = NewPendingTask("task-alpha") with { CreatedAtUtc = _time.GetUtcNow().AddMinutes(-5) };
        var middle   = NewPendingTask("task-beta")  with { CreatedAtUtc = _time.GetUtcNow().AddMinutes(-3) };
        var latest   = NewPendingTask("task-gamma") with { CreatedAtUtc = _time.GetUtcNow().AddMinutes(-1) };

        _store.EnqueuePending(earliest);
        _store.EnqueuePending(middle);
        _store.EnqueuePending(latest);

        var first = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));
        var second = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));
        var third = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));

        Assert.Equal("task-alpha", first?.TaskId);
        Assert.Equal("task-beta",  second?.TaskId);
        Assert.Equal("task-gamma", third?.TaskId);
    }

    [Fact]
    public async Task TryClaimNextPending_is_safe_under_parallel_contention()
    {
        // Enqueue exactly one task and fire N parallel claim attempts at
        // it. Only one should win; every other must return null.
        _store.EnqueuePending(NewPendingTask("task-race"));

        const int parallelism = 32;
        var winners = new ConcurrentBag<WorkTaskRecord>();

        var tasks = Enumerable.Range(0, parallelism).Select(i => Task.Run(() =>
        {
            var claimed = _store.TryClaimNextPending($"worker-{i}", TimeSpan.FromMinutes(10));
            if (claimed is not null)
            {
                winners.Add(claimed);
            }
        })).ToArray();
        await Task.WhenAll(tasks);

        Assert.Single(winners);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Assigned));
        Assert.Equal(0, _store.CountByState(WorkTaskState.Pending));
    }

    // ── completion / failure ────────────────────────────────────────

    [Fact]
    public void MarkCompleted_transitions_assigned_to_done()
    {
        _store.EnqueuePending(NewPendingTask("task-004"));
        var claimed = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));
        Assert.NotNull(claimed);

        var completed = _store.MarkCompleted("task-004", "worker-1");

        Assert.True(completed);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Done));
        Assert.Equal(0, _store.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void MarkCompleted_rejects_submission_from_wrong_worker()
    {
        _store.EnqueuePending(NewPendingTask("task-005"));
        _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));

        var completed = _store.MarkCompleted("task-005", "worker-IMPOSTER");

        Assert.False(completed);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Assigned));
        Assert.Equal(0, _store.CountByState(WorkTaskState.Done));
    }

    [Fact]
    public void MarkFailed_transitions_assigned_to_failed()
    {
        _store.EnqueuePending(NewPendingTask("task-006"));
        _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10));

        var failed = _store.MarkFailed("task-006", "worker-1");

        Assert.True(failed);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Failed));
    }

    // ── deadline recycling ──────────────────────────────────────────

    [Fact]
    public void RecycleTimedOutAssignments_returns_stale_task_to_pending()
    {
        _store.EnqueuePending(NewPendingTask("task-007"));
        var claimed = _store.TryClaimNextPending("worker-slow", TimeSpan.FromMinutes(5));
        Assert.NotNull(claimed);

        // Advance the fake clock past the deadline.
        _time.Advance(TimeSpan.FromMinutes(6));

        var recycled = _store.RecycleTimedOutAssignments();

        Assert.Equal(1, recycled);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending));
        Assert.Equal(0, _store.CountByState(WorkTaskState.Assigned));

        // The recycled task must be claimable again by a different
        // worker, and the attempt counter should have incremented.
        var reclaimed = _store.TryClaimNextPending("worker-fresh", TimeSpan.FromMinutes(5));
        Assert.NotNull(reclaimed);
        Assert.Equal(2, reclaimed!.Attempt);
        Assert.Equal("worker-fresh", reclaimed.AssignedWorkerId);
    }

    [Fact]
    public void RequeueFailedTasks_flips_every_failed_row_back_to_pending()
    {
        _store.EnqueuePending(NewPendingTask("task-fail-1"));
        _store.EnqueuePending(NewPendingTask("task-fail-2"));
        _store.EnqueuePending(NewPendingTask("task-done"));

        // Claim ordering is by created_at,task_id ASC — drain whatever
        // actually comes back rather than assuming the insertion order.
        var c1 = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10))!;
        var c2 = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10))!;
        var c3 = _store.TryClaimNextPending("worker-1", TimeSpan.FromMinutes(10))!;
        Assert.True(_store.MarkFailed(c1.TaskId, "worker-1"));
        Assert.True(_store.MarkFailed(c2.TaskId, "worker-1"));
        Assert.True(_store.MarkCompleted(c3.TaskId, "worker-1"));

        Assert.Equal(2, _store.CountByState(WorkTaskState.Failed));
        Assert.Equal(1, _store.CountByState(WorkTaskState.Done));

        var requeued = _store.RequeueFailedTasks();

        Assert.Equal(2, requeued);
        Assert.Equal(0, _store.CountByState(WorkTaskState.Failed));
        Assert.Equal(2, _store.CountByState(WorkTaskState.Pending));
        Assert.Equal(1, _store.CountByState(WorkTaskState.Done));

        // Requeued rows must be claimable again.
        var claim = _store.TryClaimNextPending("worker-fresh", TimeSpan.FromMinutes(5));
        Assert.NotNull(claim);
    }

    [Fact]
    public void RequeueFailedTasks_is_noop_when_no_failed_rows_exist()
    {
        _store.EnqueuePending(NewPendingTask("task-pending-only"));

        Assert.Equal(0, _store.RequeueFailedTasks());
        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending));
    }

    [Fact]
    public void KickStuckTasks_kicks_assigned_when_worker_heartbeat_stale()
    {
        using var workerStore = new SqliteWorkerRegistryStore(_connectionString, _time);
        workerStore.Upsert(NewWorker("worker-ghost"));

        _store.EnqueuePending(NewPendingTask("task-stuck"));
        _store.TryClaimNextPending("worker-ghost", TimeSpan.FromMinutes(5));

        // Advance time past both the lease deadline AND the stale-worker threshold.
        // Worker heartbeat was never refreshed so it is now stale relative to $cutoff.
        _time.Advance(TimeSpan.FromMinutes(10));

        var kicked = _store.KickStuckTasks(TimeSpan.FromSeconds(60));

        Assert.Equal(1, kicked);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending));
        Assert.Equal(0, _store.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void KickStuckTasks_leaves_assigned_with_fresh_heartbeat_alone()
    {
        using var workerStore = new SqliteWorkerRegistryStore(_connectionString, _time);
        workerStore.Upsert(NewWorker("worker-alive"));

        _store.EnqueuePending(NewPendingTask("task-soft"));
        _store.TryClaimNextPending("worker-alive", TimeSpan.FromMinutes(5));

        // Advance past deadline but refresh worker heartbeat so it stays alive.
        _time.Advance(TimeSpan.FromMinutes(6));
        workerStore.TouchHeartbeat("worker-alive");

        var kicked = _store.KickStuckTasks(TimeSpan.FromSeconds(60));

        Assert.Equal(0, kicked);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void KickStuckTasks_kicks_assigned_when_worker_row_missing()
    {
        // Ensure workers table exists (registry store creates the schema),
        // but leave it empty — simulates a worker that was never registered
        // or was purged after its heartbeat went stale.
        using (var _ = new SqliteWorkerRegistryStore(_connectionString, _time)) { }

        _store.EnqueuePending(NewPendingTask("task-orphan"));
        _store.TryClaimNextPending("worker-deleted", TimeSpan.FromMinutes(5));
        _time.Advance(TimeSpan.FromMinutes(10));

        var kicked = _store.KickStuckTasks(TimeSpan.FromSeconds(60));

        Assert.Equal(1, kicked);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Pending));
    }

    [Fact]
    public void RecycleTimedOutAssignments_leaves_fresh_assignments_alone()
    {
        _store.EnqueuePending(NewPendingTask("task-008"));
        _store.TryClaimNextPending("worker-quick", TimeSpan.FromMinutes(10));

        // Only move the clock by a minute — well inside the lease.
        _time.Advance(TimeSpan.FromMinutes(1));

        var recycled = _store.RecycleTimedOutAssignments();

        Assert.Equal(0, recycled);
        Assert.Equal(1, _store.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void GetById_returns_null_for_unknown_task()
    {
        Assert.Null(_store.GetById("no-such-task"));
    }

    [Fact]
    public void GetById_round_trips_the_enqueued_row()
    {
        _store.EnqueuePending(NewPendingTask("task-009"));

        var row = _store.GetById("task-009");

        Assert.NotNull(row);
        Assert.Equal("task-009", row!.TaskId);
        Assert.Equal("shard-A", row.ShardId);
        Assert.Equal(4096, row.TokensPerTask);
        Assert.Equal(WorkTaskState.Pending, row.State);
    }
}

/// <summary>
/// Simple fake <see cref="TimeProvider"/> so tests can advance the clock
/// without sleeping. Not thread-safe — tests that need cross-thread
/// clock mutation should use <see cref="Lock"/>-guarded access.
/// </summary>
internal sealed class FakeTimeProvider : TimeProvider
{
    private DateTimeOffset _now;

    public FakeTimeProvider(DateTimeOffset now)
    {
        _now = now;
    }

    public override DateTimeOffset GetUtcNow() => _now;

    public void Advance(TimeSpan by) => _now = _now.Add(by);
}
#endif
