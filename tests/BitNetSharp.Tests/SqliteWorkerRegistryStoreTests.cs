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
        string bearerTokenHash,
        WorkerState state = WorkerState.Active)
    {
        return new WorkerRecord(
            WorkerId: workerId,
            Name: name,
            BearerTokenHash: bearerTokenHash,
            CpuThreads: 16,
            TokensPerSecond: 3155d,
            RecommendedTokensPerTask: 473_600L,
            ProcessArchitecture: "X64",
            OsDescription: "Microsoft Windows 10.0.26200",
            RegisteredAtUtc: _time.GetUtcNow(),
            LastHeartbeatUtc: _time.GetUtcNow(),
            State: state);
    }

    // ── insert / find ───────────────────────────────────────────────

    [Fact]
    public void Insert_persists_row_and_round_trips_via_FindById()
    {
        var worker = NewWorker("w-001", "PAYTON-LEGION2", "hash-alpha");

        _store.Insert(worker);
        var round = _store.FindById("w-001");

        Assert.NotNull(round);
        Assert.Equal("PAYTON-LEGION2", round!.Name);
        Assert.Equal("hash-alpha", round.BearerTokenHash);
        Assert.Equal(16, round.CpuThreads);
        Assert.Equal(3155d, round.TokensPerSecond);
        Assert.Equal(473_600L, round.RecommendedTokensPerTask);
        Assert.Equal(WorkerState.Active, round.State);
    }

    [Fact]
    public void Insert_rejects_duplicate_worker_id()
    {
        _store.Insert(NewWorker("w-002", "name-a", "hash-1"));
        var duplicate = NewWorker("w-002", "name-b", "hash-2");

        Assert.Throws<InvalidOperationException>(() => _store.Insert(duplicate));
    }

    [Fact]
    public void Insert_rejects_duplicate_bearer_token_hash()
    {
        _store.Insert(NewWorker("w-003", "name-a", "hash-shared"));
        var duplicate = NewWorker("w-004", "name-b", "hash-shared");

        Assert.Throws<InvalidOperationException>(() => _store.Insert(duplicate));
    }

    [Fact]
    public void FindById_returns_null_for_unknown_worker()
    {
        Assert.Null(_store.FindById("no-such-worker"));
    }

    [Fact]
    public void FindByBearerTokenHash_returns_the_matching_worker()
    {
        _store.Insert(NewWorker("w-005", "name-a", "hash-find-me"));

        var found = _store.FindByBearerTokenHash("hash-find-me");

        Assert.NotNull(found);
        Assert.Equal("w-005", found!.WorkerId);
    }

    [Fact]
    public void FindByBearerTokenHash_returns_null_for_unknown_hash()
    {
        Assert.Null(_store.FindByBearerTokenHash("hash-does-not-exist"));
    }

    // ── heartbeat ───────────────────────────────────────────────────

    [Fact]
    public void TouchHeartbeat_updates_last_heartbeat_timestamp()
    {
        _store.Insert(NewWorker("w-006", "name-a", "hash-a"));

        _time.Advance(TimeSpan.FromMinutes(5));
        var touched = _store.TouchHeartbeat("w-006");

        Assert.True(touched);
        var updated = _store.FindById("w-006");
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
        _store.Insert(NewWorker("w-007", "name-a", "hash-a"));

        Assert.True(_store.MarkDraining("w-007"));

        var after = _store.FindById("w-007");
        Assert.Equal(WorkerState.Draining, after!.State);
    }

    [Fact]
    public void MarkGone_transitions_active_to_gone()
    {
        _store.Insert(NewWorker("w-008", "name-a", "hash-a"));

        Assert.True(_store.MarkGone("w-008"));

        var after = _store.FindById("w-008");
        Assert.Equal(WorkerState.Gone, after!.State);
    }

    // ── sweep stale workers ────────────────────────────────────────

    [Fact]
    public void SweepStaleWorkers_moves_silent_actives_to_gone()
    {
        _store.Insert(NewWorker("w-009", "silent", "hash-silent"));
        _store.Insert(NewWorker("w-010", "chatty", "hash-chatty"));

        // Advance time and heartbeat only the chatty worker.
        _time.Advance(TimeSpan.FromMinutes(10));
        _store.TouchHeartbeat("w-010");

        // Another 5 minutes pass; silent worker is now ~15 min stale.
        _time.Advance(TimeSpan.FromMinutes(5));

        var swept = _store.SweepStaleWorkers(TimeSpan.FromMinutes(10));

        Assert.Equal(1, swept);
        Assert.Equal(WorkerState.Gone, _store.FindById("w-009")!.State);
        Assert.Equal(WorkerState.Active, _store.FindById("w-010")!.State);
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
        _store.Insert(NewWorker("w-011", "a", "h1"));
        _store.Insert(NewWorker("w-012", "b", "h2"));
        _store.Insert(NewWorker("w-013", "c", "h3"));
        _store.MarkDraining("w-012");
        _store.MarkGone("w-013");

        Assert.Equal(1, _store.CountByState(WorkerState.Active));
        Assert.Equal(1, _store.CountByState(WorkerState.Draining));
        Assert.Equal(1, _store.CountByState(WorkerState.Gone));
    }

    [Fact]
    public void ListAll_returns_rows_in_registration_order()
    {
        _store.Insert(NewWorker("w-first",  "alpha", "h-first"));
        _time.Advance(TimeSpan.FromSeconds(1));
        _store.Insert(NewWorker("w-second", "beta",  "h-second") with { RegisteredAtUtc = _time.GetUtcNow() });
        _time.Advance(TimeSpan.FromSeconds(1));
        _store.Insert(NewWorker("w-third",  "gamma", "h-third") with { RegisteredAtUtc = _time.GetUtcNow() });

        var all = _store.ListAll();

        Assert.Equal(3, all.Count);
        Assert.Equal("w-first",  all[0].WorkerId);
        Assert.Equal("w-second", all[1].WorkerId);
        Assert.Equal("w-third",  all[2].WorkerId);
    }
}
