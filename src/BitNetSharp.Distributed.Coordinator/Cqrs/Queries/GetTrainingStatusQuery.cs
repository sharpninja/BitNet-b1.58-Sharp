using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using McpServer.Cqrs;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Queries;

/// <summary>
/// Query behind the admin <c>/admin/training-status</c> page. Returns a
/// single snapshot that bundles the per-shard-prefix rollup, the
/// per-worker × per-shard grid, the fleet + per-worker sparklines, and
/// the recent gradient-event feed so the page can render top-to-bottom
/// without fanning out multiple server calls.
/// </summary>
public sealed class GetTrainingStatusQuery : IQuery<TrainingStatusSnapshot>
{
    /// <summary>Size of the recent-events feed.</summary>
    public int RecentEventsLimit { get; init; } = 20;

    /// <summary>Length of each sparkline in buckets.</summary>
    public int SparklineBucketCount { get; init; } = 60;

    /// <summary>Width of a single sparkline bucket.</summary>
    public TimeSpan SparklineBucketSize { get; init; } = TimeSpan.FromMinutes(1);

    /// <summary>Upper bound on per-worker sparklines (payload-size guard).</summary>
    public int MaxPerWorkerSparklines { get; init; } = 12;
}

public sealed record PrefixRollupRow(
    string Prefix,
    string DisplayLabel,
    int Queued,
    int Assigned,
    int Done,
    int Failed,
    double PercentComplete,
    double? EtaSeconds,
    double TokensPerSecond);

public sealed record WorkerShardCell(
    string ClientId,
    string ShardPrefix,
    long TasksCompleted,
    double TokensPerSecond);

public sealed record SparklineSeries(
    string Key,
    IReadOnlyList<double> TokensPerSecondBuckets,
    DateTimeOffset FirstBucketUtc,
    TimeSpan BucketSize);

public sealed record TrainingStatusSnapshot(
    long CurrentWeightVersion,
    double GlobalTokensPerSecond,
    int ActiveWorkerCount,
    double? FleetEtaSeconds,
    IReadOnlyList<PrefixRollupRow> Rollups,
    IReadOnlyList<WorkerShardCell> WorkerShardCells,
    IReadOnlyList<SparklineSeries> Sparklines,
    IReadOnlyList<RecentGradientEvent> RecentEvents,
    DateTimeOffset GeneratedAtUtc);

public sealed class GetTrainingStatusQueryHandler
    : IQueryHandler<GetTrainingStatusQuery, TrainingStatusSnapshot>
{
    /// <summary>
    /// Average-task size estimate used to turn remaining-task count into
    /// seconds-to-drain. Matches the coordinator's default task sizing
    /// (see Program.cs tokens-per-task fallback).
    /// </summary>
    public const long DefaultTokensPerTask = 16_384;

    /// <summary>Window the live tok/s header reads over.</summary>
    public static readonly TimeSpan LiveWindow = TimeSpan.FromMinutes(1);

    private readonly SqliteWorkQueueStore _workQueue;
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly WeightApplicationService _weights;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly TimeProvider _time;

    public GetTrainingStatusQueryHandler(
        SqliteWorkQueueStore workQueue,
        SqliteWorkerRegistryStore workerStore,
        SqliteTelemetryStore telemetry,
        WeightApplicationService weights,
        IOptionsMonitor<CoordinatorOptions> options,
        TimeProvider time)
    {
        _workQueue = workQueue;
        _workerStore = workerStore;
        _telemetry = telemetry;
        _weights = weights;
        _options = options;
        _time = time;
    }

    public Task<Result<TrainingStatusSnapshot>> HandleAsync(
        GetTrainingStatusQuery query,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(query);
        var now = _time.GetUtcNow();
        var opts = _options.CurrentValue;
        var prefixes = opts.ActiveShardPrefixes ?? Array.Empty<ShardPrefixConfig>();

        var liveStart = now - LiveWindow;
        var globalLive = _telemetry.AggregateGlobal(liveStart);
        var globalTps = LiveWindow.TotalSeconds > 0d
            ? globalLive.TokensSeen / LiveWindow.TotalSeconds
            : 0d;

        var activeWorkers = _workerStore.CountByState(WorkerState.Active);

        // Per-worker × per-shard aggregate feeds both the grid and the
        // per-prefix tok/s sum in rollup rows.
        var prefixStrings = prefixes.Select(p => p.Prefix).ToArray();
        var workerShardAggregates = _telemetry.AggregateByWorkerAndShardPrefix(
            liveStart, prefixStrings);

        var rollups = new List<PrefixRollupRow>(prefixes.Count);
        foreach (var p in prefixes)
        {
            var queued = _workQueue.CountByShardPrefixAndState(p.Prefix, WorkTaskState.Pending);
            var assigned = _workQueue.CountByShardPrefixAndState(p.Prefix, WorkTaskState.Assigned);
            var done = _workQueue.CountByShardPrefixAndState(p.Prefix, WorkTaskState.Done);
            var failed = _workQueue.CountByShardPrefixAndState(p.Prefix, WorkTaskState.Failed);
            var total = queued + assigned + done + failed;
            var pct = total > 0 ? 100.0 * done / total : 0d;

            var prefixTokens = workerShardAggregates
                .Where(a => a.ShardPrefix == p.Prefix)
                .Sum(a => a.TokensSeen);
            var prefixTps = LiveWindow.TotalSeconds > 0d
                ? prefixTokens / LiveWindow.TotalSeconds
                : 0d;

            var remaining = (long)(queued + assigned);
            double? eta = null;
            if (prefixTps > 0d && remaining > 0)
            {
                eta = remaining * DefaultTokensPerTask / prefixTps;
            }

            rollups.Add(new PrefixRollupRow(
                Prefix: p.Prefix,
                DisplayLabel: p.DisplayLabel,
                Queued: queued,
                Assigned: assigned,
                Done: done,
                Failed: failed,
                PercentComplete: pct,
                EtaSeconds: eta,
                TokensPerSecond: prefixTps));
        }

        // Fleet ETA from total remaining ASR + TruckMate real-work rows.
        var totalRemaining = rollups.Sum(r => (long)(r.Queued + r.Assigned));
        double? fleetEta = null;
        if (globalTps > 0d && totalRemaining > 0)
        {
            fleetEta = totalRemaining * DefaultTokensPerTask / globalTps;
        }

        // Grid cells: emit one per (client, prefix) with a non-zero count.
        var cells = workerShardAggregates.Select(a => new WorkerShardCell(
            ClientId: a.ClientId,
            ShardPrefix: a.ShardPrefix,
            TasksCompleted: a.TasksCompleted,
            TokensPerSecond: LiveWindow.TotalSeconds > 0d
                ? a.TokensSeen / LiveWindow.TotalSeconds
                : 0d)).ToArray();

        // Sparklines: one fleet series + top-N active-worker series,
        // where "top-N" is ranked by recent tokens seen in the live
        // window so the busiest workers get visualised first.
        var sparklines = new List<SparklineSeries>();
        var sparkStart = now - query.SparklineBucketSize * query.SparklineBucketCount;
        sparklines.Add(BuildSeries("fleet", sparkStart, query.SparklineBucketSize,
            query.SparklineBucketCount, clientId: null));

        var topWorkers = workerShardAggregates
            .GroupBy(a => a.ClientId)
            .Select(g => new { ClientId = g.Key, Tokens = g.Sum(x => x.TokensSeen) })
            .OrderByDescending(x => x.Tokens)
            .Take(query.MaxPerWorkerSparklines)
            .Select(x => x.ClientId)
            .ToArray();
        foreach (var clientId in topWorkers)
        {
            sparklines.Add(BuildSeries(clientId, sparkStart, query.SparklineBucketSize,
                query.SparklineBucketCount, clientId));
        }

        var recent = _telemetry.GetRecentGradientEvents(query.RecentEventsLimit);

        var snapshot = new TrainingStatusSnapshot(
            CurrentWeightVersion: _weights.CurrentVersion,
            GlobalTokensPerSecond: globalTps,
            ActiveWorkerCount: activeWorkers,
            FleetEtaSeconds: fleetEta,
            Rollups: rollups,
            WorkerShardCells: cells,
            Sparklines: sparklines,
            RecentEvents: recent,
            GeneratedAtUtc: now);

        return Task.FromResult(Result<TrainingStatusSnapshot>.Success(snapshot));
    }

    private SparklineSeries BuildSeries(
        string key,
        DateTimeOffset since,
        TimeSpan bucketSize,
        int bucketCount,
        string? clientId)
    {
        var raw = _telemetry.GetThroughputBuckets(since, bucketSize, clientId);
        var byStart = raw.ToDictionary(b => b.BucketStartUtc.ToUnixTimeSeconds(), b => b.TokensSeen);
        var bucketSec = (long)bucketSize.TotalSeconds;
        if (bucketSec <= 0) bucketSec = 1;
        var firstBucket = new DateTimeOffset((since.ToUnixTimeSeconds() / bucketSec) * bucketSec * TimeSpan.TicksPerSecond + DateTimeOffset.UnixEpoch.Ticks, TimeSpan.Zero);

        var values = new double[bucketCount];
        for (var i = 0; i < bucketCount; i++)
        {
            var bStart = firstBucket.AddSeconds(i * bucketSec);
            if (byStart.TryGetValue(bStart.ToUnixTimeSeconds(), out var tokens))
            {
                values[i] = bucketSize.TotalSeconds > 0
                    ? tokens / bucketSize.TotalSeconds
                    : 0d;
            }
            else
            {
                values[i] = 0d;
            }
        }

        return new SparklineSeries(
            Key: key,
            TokensPerSecondBuckets: values,
            FirstBucketUtc: firstBucket,
            BucketSize: bucketSize);
    }
}
