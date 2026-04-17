using System;
using System.IO;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Tests the single-iteration sweep path of
/// <see cref="StaleSweeperService"/>. Background service wiring
/// itself is exercised integration-style when the full coordinator
/// harness lands; here we just confirm one tick does the right
/// thing against the real stores.
/// </summary>
public sealed class StaleSweeperServiceTests : IDisposable
{
    private readonly string _databasePath;
    private readonly FakeTimeProvider _time;
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteWorkQueueStore _queueStore;
    private readonly StaleSweeperService _service;

    public StaleSweeperServiceTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-sweeper-{Guid.NewGuid():N}.db");
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 20, 0, 0, TimeSpan.Zero));
        var connectionString = $"Data Source={_databasePath}";
        _workerStore = new SqliteWorkerRegistryStore(connectionString, _time);
        _queueStore = new SqliteWorkQueueStore(connectionString, _time);

        var options = Options.Create(new CoordinatorOptions
        {
            StaleWorkerThresholdSeconds = 60
        });
        var monitor = new StaticOptionsMonitor<CoordinatorOptions>(options.Value);
        _service = new StaleSweeperService(
            _workerStore,
            _queueStore,
            monitor,
            NullLogger<StaleSweeperService>.Instance,
            _time);
    }

    public void Dispose()
    {
        _workerStore.Dispose();
        _queueStore.Dispose();
        TryDelete(_databasePath);
        TryDelete(_databasePath + "-wal");
        TryDelete(_databasePath + "-shm");
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
    }

    private WorkerRecord NewActiveWorker(string id) => new(
        WorkerId: id,
        Name: id,
        CpuThreads: 4,
        TokensPerSecond: 1000d,
        RecommendedTokensPerTask: 150_016L,
        ProcessArchitecture: "X64",
        OsDescription: "TestOS",
        RegisteredAtUtc: _time.GetUtcNow(),
        LastHeartbeatUtc: _time.GetUtcNow(),
        State: WorkerState.Active);

    private WorkTaskRecord NewPendingTask(string id) => new(
        TaskId: id,
        WeightVersion: 1,
        ShardId: "shard-A",
        ShardOffset: 0,
        ShardLength: 1024,
        TokensPerTask: 4096,
        KLocalSteps: 4,
        HyperparametersJson: "{}",
        State: WorkTaskState.Pending,
        AssignedWorkerId: null,
        AssignedAtUtc: null,
        DeadlineUtc: null,
        Attempt: 0,
        CreatedAtUtc: _time.GetUtcNow(),
        CompletedAtUtc: null);

    [Fact]
    public void SweepOnce_on_empty_state_returns_zero_both_counts()
    {
        var result = _service.SweepOnce();
        Assert.Equal(new SweepResult(0, 0), result);
    }

    [Fact]
    public void SweepOnce_transitions_stale_workers_to_gone()
    {
        _workerStore.Upsert(NewActiveWorker("worker-silent"));
        _workerStore.Upsert(NewActiveWorker("worker-chatty"));

        // Advance past the stale threshold; only the chatty worker
        // keeps its heartbeat fresh.
        _time.Advance(TimeSpan.FromSeconds(90));
        _workerStore.TouchHeartbeat("worker-chatty");

        var result = _service.SweepOnce();

        Assert.Equal(1, result.WorkersSweptToGone);
        Assert.Equal(WorkerState.Gone, _workerStore.FindById("worker-silent")!.State);
        Assert.Equal(WorkerState.Active, _workerStore.FindById("worker-chatty")!.State);
    }

    [Fact]
    public void SweepOnce_recycles_timed_out_tasks_back_to_pending()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-timeout"));
        var claimed = _queueStore.TryClaimNextPending("worker-1", TimeSpan.FromSeconds(30));
        Assert.NotNull(claimed);

        // Advance past the task deadline.
        _time.Advance(TimeSpan.FromSeconds(45));

        var result = _service.SweepOnce();

        Assert.Equal(1, result.TasksRecycled);
        Assert.Equal(1, _queueStore.CountByState(WorkTaskState.Pending));
        Assert.Equal(0, _queueStore.CountByState(WorkTaskState.Assigned));
    }

    [Fact]
    public void SweepOnce_returns_combined_counts_when_both_sides_have_work()
    {
        _workerStore.Upsert(NewActiveWorker("worker-silent"));
        _queueStore.EnqueuePending(NewPendingTask("task-stuck"));
        _queueStore.TryClaimNextPending("worker-1", TimeSpan.FromSeconds(30));

        _time.Advance(TimeSpan.FromSeconds(120));

        var result = _service.SweepOnce();

        Assert.Equal(1, result.WorkersSweptToGone);
        Assert.Equal(1, result.TasksRecycled);
    }
}

/// <summary>
/// Trivial <see cref="IOptionsMonitor{T}"/> that always returns the
/// static value it was constructed with. Good enough for the
/// sweeper's configuration reads; does not support change tokens.
/// </summary>
internal sealed class StaticOptionsMonitor<T> : IOptionsMonitor<T>
{
    private readonly T _value;

    public StaticOptionsMonitor(T value) { _value = value; }

    public T CurrentValue => _value;

    public T Get(string? name) => _value;

    public IDisposable? OnChange(Action<T, string?> listener) => null;
}
