using System;
using System.IO;
using System.Linq;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for the three training-status store methods on
/// <see cref="SqliteTelemetryStore"/>: recent-events feed,
/// per-worker-per-shard-prefix rollup, and time-bucketed throughput
/// sparkline source. Uses a shared temp DB so the telemetry store can
/// join against the <see cref="SqliteWorkQueueStore"/>-owned
/// <c>tasks</c> table.
/// </summary>
public sealed class SqliteTelemetryStoreTests : IDisposable
{
    private readonly string _databasePath;
    private readonly string _connectionString;
    private readonly FakeTimeProvider _time;
    private readonly SqliteWorkQueueStore _queueStore;
    private readonly SqliteTelemetryStore _telemetry;

    public SqliteTelemetryStoreTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-tele-{Guid.NewGuid():N}.db");
        _connectionString = $"Data Source={_databasePath}";
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 18, 0, 0, TimeSpan.Zero));
        _queueStore = new SqliteWorkQueueStore(_connectionString, _time);
        _telemetry = new SqliteTelemetryStore(_connectionString, _time);
    }

    public void Dispose()
    {
        _telemetry.Dispose();
        _queueStore.Dispose();
        TryDelete(_databasePath);
        TryDelete(_databasePath + "-wal");
        TryDelete(_databasePath + "-shm");
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
    }

    private void EnqueueTask(string taskId, string shardId)
    {
        _queueStore.EnqueuePending(new WorkTaskRecord(
            TaskId: taskId,
            WeightVersion: 1,
            ShardId: shardId,
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
            CompletedAtUtc: null));
    }

    // ── GetRecentGradientEvents ────────────────────────────────────

    [Fact]
    public void GetRecentGradientEvents_joins_shard_id_from_tasks()
    {
        EnqueueTask("t-1", "asr-v1-shard-7");
        _telemetry.RecordAccepted("w-1", "t-1", 1000, 500, 0, 0.1f, 2, 1.0);

        var recent = _telemetry.GetRecentGradientEvents(10);

        var row = Assert.Single(recent);
        Assert.Equal("t-1", row.TaskId);
        Assert.Equal("asr-v1-shard-7", row.ShardId);
        Assert.Equal("w-1", row.ClientId);
    }

    [Fact]
    public void GetRecentGradientEvents_orphan_event_returns_empty_shard_id()
    {
        // Task row never enqueued — event is an orphan.
        _telemetry.RecordAccepted("w-1", "t-orphan", 1000, 500, 0, 0.1f, 2, 1.0);

        var recent = _telemetry.GetRecentGradientEvents(10);

        var row = Assert.Single(recent);
        Assert.Equal("t-orphan", row.TaskId);
        Assert.Equal(string.Empty, row.ShardId);
    }

    [Fact]
    public void GetRecentGradientEvents_orders_newest_first_and_respects_limit()
    {
        for (var i = 0; i < 5; i++)
        {
            EnqueueTask($"t-{i}", $"asr-v1-shard-{i}");
            _telemetry.RecordAccepted("w-1", $"t-{i}", 100, 50, 0, 0.1f, i + 1, 1.0);
            _time.Advance(TimeSpan.FromSeconds(1));
        }

        var recent = _telemetry.GetRecentGradientEvents(3);

        Assert.Equal(3, recent.Count);
        Assert.Equal("t-4", recent[0].TaskId);
        Assert.Equal("t-3", recent[1].TaskId);
        Assert.Equal("t-2", recent[2].TaskId);
    }

    [Fact]
    public void GetRecentGradientEvents_returns_empty_for_zero_limit()
    {
        EnqueueTask("t-1", "asr-v1-shard-0");
        _telemetry.RecordAccepted("w-1", "t-1", 1000, 500, 0, 0.1f, 2, 1.0);

        Assert.Empty(_telemetry.GetRecentGradientEvents(0));
    }

    // ── AggregateByWorkerAndShardPrefix ────────────────────────────

    [Fact]
    public void AggregateByWorkerAndShardPrefix_groups_by_prefix_and_worker()
    {
        EnqueueTask("t-asr-1", "asr-v1-shard-0");
        EnqueueTask("t-asr-2", "asr-v1-shard-1");
        EnqueueTask("t-tm-1", "truckmate-v2-shard-0");

        _telemetry.RecordAccepted("w-1", "t-asr-1", 100, 10, 0, 0.1f, 1, 1.0);
        _telemetry.RecordAccepted("w-1", "t-asr-2", 200, 20, 0, 0.1f, 2, 1.0);
        _telemetry.RecordAccepted("w-2", "t-tm-1",  500, 50, 0, 0.1f, 3, 1.0);

        var since = _time.GetUtcNow().AddMinutes(-5);
        var rows = _telemetry.AggregateByWorkerAndShardPrefix(
            since, new[] { "asr-v1-", "truckmate-v2-" });

        Assert.Equal(2, rows.Count);
        var asr = rows.Single(r => r.ShardPrefix == "asr-v1-");
        Assert.Equal("w-1", asr.ClientId);
        Assert.Equal(2, asr.TasksCompleted);
        Assert.Equal(300, asr.TokensSeen);

        var tm = rows.Single(r => r.ShardPrefix == "truckmate-v2-");
        Assert.Equal("w-2", tm.ClientId);
        Assert.Equal(1, tm.TasksCompleted);
        Assert.Equal(500, tm.TokensSeen);
    }

    [Fact]
    public void AggregateByWorkerAndShardPrefix_ignores_events_outside_window()
    {
        EnqueueTask("t-old", "asr-v1-shard-0");
        _telemetry.RecordAccepted("w-1", "t-old", 999, 100, 0, 0.1f, 1, 1.0);

        _time.Advance(TimeSpan.FromHours(2));

        EnqueueTask("t-new", "asr-v1-shard-1");
        _telemetry.RecordAccepted("w-1", "t-new", 100, 10, 0, 0.1f, 2, 1.0);

        var since = _time.GetUtcNow().AddMinutes(-5);
        var rows = _telemetry.AggregateByWorkerAndShardPrefix(since, new[] { "asr-v1-" });

        var row = Assert.Single(rows);
        Assert.Equal(1, row.TasksCompleted);
        Assert.Equal(100, row.TokensSeen);
    }

    [Fact]
    public void AggregateByWorkerAndShardPrefix_drops_orphan_events()
    {
        // Event with no tasks row — join should drop it.
        _telemetry.RecordAccepted("w-1", "t-orphan", 100, 10, 0, 0.1f, 1, 1.0);

        var since = _time.GetUtcNow().AddMinutes(-5);
        var rows = _telemetry.AggregateByWorkerAndShardPrefix(since, new[] { "asr-v1-" });

        Assert.Empty(rows);
    }

    [Fact]
    public void AggregateByWorkerAndShardPrefix_empty_prefix_list_returns_empty()
    {
        Assert.Empty(_telemetry.AggregateByWorkerAndShardPrefix(
            _time.GetUtcNow().AddMinutes(-5), Array.Empty<string>()));
    }

    // ── GetThroughputBuckets ───────────────────────────────────────

    [Fact]
    public void GetThroughputBuckets_buckets_events_into_window()
    {
        _telemetry.RecordAccepted("w-1", "t-a", 100, 10, 0, 0.1f, 1, 1.0);
        _telemetry.RecordAccepted("w-1", "t-b", 200, 20, 0, 0.1f, 2, 1.0);

        // Advance 1 bucket, record another event in the next bucket.
        _time.Advance(TimeSpan.FromMinutes(1));
        _telemetry.RecordAccepted("w-1", "t-c", 300, 30, 0, 0.1f, 3, 1.0);

        var since = _time.GetUtcNow().AddMinutes(-5);
        var buckets = _telemetry.GetThroughputBuckets(since, TimeSpan.FromMinutes(1));

        Assert.Equal(2, buckets.Count);
        Assert.Equal(300, buckets[0].TokensSeen);  // 100+200
        Assert.Equal(300, buckets[1].TokensSeen);
    }

    [Fact]
    public void GetThroughputBuckets_filters_by_client_when_supplied()
    {
        _telemetry.RecordAccepted("w-1", "t-1", 100, 10, 0, 0.1f, 1, 1.0);
        _telemetry.RecordAccepted("w-2", "t-2", 999, 10, 0, 0.1f, 2, 1.0);

        var since = _time.GetUtcNow().AddMinutes(-5);
        var buckets = _telemetry.GetThroughputBuckets(
            since, TimeSpan.FromMinutes(1), clientId: "w-1");

        var row = Assert.Single(buckets);
        Assert.Equal(100, row.TokensSeen);
    }

    [Fact]
    public void GetThroughputBuckets_emits_no_gaps_only_buckets_with_events()
    {
        _telemetry.RecordAccepted("w-1", "t-a", 100, 10, 0, 0.1f, 1, 1.0);

        // Skip 3 buckets without any event.
        _time.Advance(TimeSpan.FromMinutes(3));
        _telemetry.RecordAccepted("w-1", "t-b", 200, 20, 0, 0.1f, 2, 1.0);

        var since = _time.GetUtcNow().AddMinutes(-10);
        var buckets = _telemetry.GetThroughputBuckets(since, TimeSpan.FromMinutes(1));

        // Two event-bearing buckets; empty intermediate buckets are NOT emitted.
        Assert.Equal(2, buckets.Count);
    }

    [Fact]
    public void GetThroughputBuckets_throws_on_non_positive_bucket_size()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _telemetry.GetThroughputBuckets(_time.GetUtcNow(), TimeSpan.Zero));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            _telemetry.GetThroughputBuckets(_time.GetUtcNow(), TimeSpan.FromSeconds(-1)));
    }
}
