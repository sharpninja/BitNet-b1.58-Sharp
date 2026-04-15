#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests covering every public surface of
/// <see cref="SqliteWorkerRegistryStore"/>. A fresh temp file per test
/// class instance isolates WAL state so parallel xunit runs never stomp
/// on each other.
/// </summary>
public sealed class SqliteWorkerRegistryStoreTests : IDisposable
{
    private readonly string _databasePath;
    private readonly FakeTimeProvider _time;
    private readonly SqliteWorkerRegistryStore _store;

    public SqliteWorkerRegistryStoreTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-wreg-{Guid.NewGuid():N}.db");
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 18, 0, 0, TimeSpan.Zero));
        _store = new SqliteWorkerRegistryStore($"Data Source={_databasePath}", _time);
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

    private WorkerRecord NewWorker(
        string workerId,
        string name,
        WorkerState state = WorkerState.Active,
        double tokensPerSecond = 3155d,
        long recommended = 473_600L)
    {
        return new WorkerRecord(
            WorkerId: workerId,
            Name: name,
            CpuThreads: 16,
            TokensPerSecond: tokensPerSecond,
            RecommendedTokensPerTask: recommended,
            ProcessArchitecture: "X64",
            OsDescription: "Microsoft Windows 10.0.26200",
            RegisteredAtUtc: _time.GetUtcNow(),
            LastHeartbeatUtc: _time.GetUtcNow(),
            State: state);
    }

    // ── upsert / find ───────────────────────────────────────────────

    [Fact]
    public void Upsert_persists_row_and_round_trips_via_FindById()
    {
        var worker = NewWorker("worker-alpha", "PAYTON-LEGION2");

        _store.Upsert(worker);
        var round = _store.FindById("worker-alpha");

        Assert.NotNull(round);
        Assert.Equal("PAYTON-LEGION2", round!.Name);
        Assert.Equal(16, round.CpuThreads);
        Assert.Equal(3155d, round.TokensPerSecond);
        Assert.Equal(473_600L, round.RecommendedTokensPerTask);
        Assert.Equal(WorkerState.Active, round.State);
    }

    [Fact]
    public void Upsert_updates_existing_row_on_reregistration()
    {
        var original = NewWorker("worker-beta", "legacy-name", tokensPerSecond: 1000d, recommended: 150_016L);
        _store.Upsert(original);

        var reregistered = NewWorker("worker-beta", "fresh-name", tokensPerSecond: 4000d, recommended: 600_064L);
        _store.Upsert(reregistered);

        var latest = _store.FindById("worker-beta");
        Assert.NotNull(latest);
        Assert.Equal("fresh-name", latest!.Name);
        Assert.Equal(4000d, latest.TokensPerSecond);
        Assert.Equal(600_064L, latest.RecommendedTokensPerTask);
    }

    [Fact]
    public void Upsert_resets_gone_worker_to_active_on_reregistration()
    {
        _store.Upsert(NewWorker("worker-revive", "original"));
        _store.MarkGone("worker-revive");
        Assert.Equal(WorkerState.Gone, _store.FindById("worker-revive")!.State);

        _store.Upsert(NewWorker("worker-revive", "revived"));

        Assert.Equal(WorkerState.Active, _store.FindById("worker-revive")!.State);
    }

    [Fact]
    public void FindById_returns_null_for_unknown_worker()
    {
        Assert.Null(_store.FindById("no-such-worker"));
    }

    // ── heartbeat ───────────────────────────────────────────────────

    [Fact]
    public void TouchHeartbeat_updates_last_heartbeat_timestamp()
    {
        _store.Upsert(NewWorker("worker-heartbeat", "name-a"));

        _time.Advance(TimeSpan.FromMinutes(5));
        var touched = _store.TouchHeartbeat("worker-heartbeat");

        Assert.True(touched);
        var updated = _store.FindById("worker-heartbeat");
        Assert.Equal(_time.GetUtcNow().ToUnixTimeSeconds(), updated!.LastHeartbeatUtc.ToUnixTimeSeconds());
    }

    [Fact]
    public void TouchHeartbeat_returns_false_for_unknown_worker()
    {
        Assert.False(_store.TouchHeartbeat("nope"));
    }

    // ── state transitions ──────────────────────────────────────────

    [Fact]
    public void MarkDraining_transitions_active_to_draining()
    {
        _store.Upsert(NewWorker("worker-drain", "name-a"));

        Assert.True(_store.MarkDraining("worker-drain"));

        var after = _store.FindById("worker-drain");
        Assert.Equal(WorkerState.Draining, after!.State);
    }

    [Fact]
    public void MarkGone_transitions_active_to_gone()
    {
        _store.Upsert(NewWorker("worker-gone", "name-a"));

        Assert.True(_store.MarkGone("worker-gone"));

        var after = _store.FindById("worker-gone");
        Assert.Equal(WorkerState.Gone, after!.State);
    }

    // ── sweep stale workers ────────────────────────────────────────

    [Fact]
    public void SweepStaleWorkers_moves_silent_actives_to_gone()
    {
        _store.Upsert(NewWorker("worker-silent", "silent"));
        _store.Upsert(NewWorker("worker-chatty", "chatty"));

        // Advance time and heartbeat only the chatty worker.
        _time.Advance(TimeSpan.FromMinutes(10));
        _store.TouchHeartbeat("worker-chatty");

        // Another 5 minutes pass; silent worker is now ~15 min stale.
        _time.Advance(TimeSpan.FromMinutes(5));

        var swept = _store.SweepStaleWorkers(TimeSpan.FromMinutes(10));

        Assert.Equal(1, swept);
        Assert.Equal(WorkerState.Gone, _store.FindById("worker-silent")!.State);
        Assert.Equal(WorkerState.Active, _store.FindById("worker-chatty")!.State);
    }

    [Fact]
    public void SweepStaleWorkers_rejects_non_positive_threshold()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => _store.SweepStaleWorkers(TimeSpan.Zero));
        Assert.Throws<ArgumentOutOfRangeException>(() => _store.SweepStaleWorkers(TimeSpan.FromSeconds(-1)));
    }

    // ── counts / listing ───────────────────────────────────────────

    [Fact]
    public void CountByState_reflects_current_lifecycle_distribution()
    {
        _store.Upsert(NewWorker("worker-11", "a"));
        _store.Upsert(NewWorker("worker-12", "b"));
        _store.Upsert(NewWorker("worker-13", "c"));
        _store.MarkDraining("worker-12");
        _store.MarkGone("worker-13");

        Assert.Equal(1, _store.CountByState(WorkerState.Active));
        Assert.Equal(1, _store.CountByState(WorkerState.Draining));
        Assert.Equal(1, _store.CountByState(WorkerState.Gone));
    }

    [Fact]
    public void ListAll_returns_rows_in_registration_order()
    {
        _store.Upsert(NewWorker("worker-first",  "alpha"));
        _time.Advance(TimeSpan.FromSeconds(1));
        _store.Upsert(NewWorker("worker-second", "beta") with { RegisteredAtUtc = _time.GetUtcNow() });
        _time.Advance(TimeSpan.FromSeconds(1));
        _store.Upsert(NewWorker("worker-third",  "gamma") with { RegisteredAtUtc = _time.GetUtcNow() });

        var all = _store.ListAll();

        Assert.Equal(3, all.Count);
        Assert.Equal("worker-first",  all[0].WorkerId);
        Assert.Equal("worker-second", all[1].WorkerId);
        Assert.Equal("worker-third",  all[2].WorkerId);
    }
}
#endif
