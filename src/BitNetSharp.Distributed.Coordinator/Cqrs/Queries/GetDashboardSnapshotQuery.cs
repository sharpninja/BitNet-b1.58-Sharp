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
/// Query that assembles every piece of data the admin dashboard
/// page needs into a single <see cref="DashboardSnapshot"/>.
/// Called on every page refresh so the operator sees current state.
/// Two rollup windows are computed: a long "Recent" window (15 min)
/// for stable averages, and a short "Live" window (1 min) for
/// responsive throughput / ETA numbers.
/// </summary>
public sealed class GetDashboardSnapshotQuery : IQuery<DashboardSnapshot>
{
}

/// <summary>
/// Point-in-time snapshot of coordinator + worker health + recent
/// throughput for the dashboard page.
/// </summary>
public sealed record DashboardSnapshot(
    long CurrentWeightVersion,
    int WeightDimension,
    TaskCounts Tasks,
    WorkerCounts Workers,
    GlobalTelemetryAggregate RecentGlobalTelemetry,
    GlobalTelemetryAggregate LiveGlobalTelemetry,
    FleetProgress Progress,
    IReadOnlyList<DashboardWorkerRow> WorkerRows,
    DateTimeOffset GeneratedAtUtc);

/// <summary>
/// Task counter rollup. <paramref name="SoftExpiredButAlive"/> counts
/// Assigned tasks whose lease deadline has passed but whose worker is
/// still sending heartbeats — a "slow but alive" signal distinct from a
/// stuck worker whose heartbeat is also stale.
/// </summary>
public sealed record TaskCounts(
    int Pending,
    int Assigned,
    int Done,
    int Failed,
    int SoftExpiredButAlive,
    int StuckDead);

public sealed record WorkerCounts(int Configured, int Active, int Draining, int Gone);

/// <summary>
/// Fleet-level progress + ETA indicators computed from the short
/// "Live" window so the dashboard reflects the current completion
/// rate instead of a long trailing average.
/// </summary>
/// <param name="TotalTasks">Pending + Assigned + Done + Failed.</param>
/// <param name="CompletedTasks">Done (does not include Failed).</param>
/// <param name="RemainingTasks">Pending + Assigned.</param>
/// <param name="PercentComplete">0..1 ratio, or 0 when TotalTasks = 0.</param>
/// <param name="TasksPerSecondLive">Tasks completed / LiveWindowSeconds over the short window.</param>
/// <param name="TokensPerSecondLive">Tokens seen / LiveWindowSeconds over the short window.</param>
/// <param name="EtaSeconds">RemainingTasks / TasksPerSecondLive; null when the live throughput is zero.</param>
/// <param name="EtaUtc">GeneratedAt + EtaSeconds; null when ETA is unknown.</param>
/// <param name="LiveWindowSeconds">Width of the short window used for the live rates.</param>
public sealed record FleetProgress(
    int TotalTasks,
    int CompletedTasks,
    int RemainingTasks,
    double PercentComplete,
    double TasksPerSecondLive,
    double TokensPerSecondLive,
    double? EtaSeconds,
    DateTimeOffset? EtaUtc,
    double LiveWindowSeconds);

/// <summary>
/// One row on the dashboard's worker table. Combines the static
/// registry info (id, name, cpu threads) with the 15-min recent
/// telemetry rollup and a 1-min live rollup so the dashboard can
/// show both smoothed averages and up-to-the-second throughput.
/// </summary>
public sealed record DashboardWorkerRow(
    string ClientId,
    string Name,
    WorkerState State,
    int CpuThreads,
    double TokensPerSecond,
    long RecommendedTokensPerTask,
    DateTimeOffset RegisteredAtUtc,
    DateTimeOffset LastHeartbeatUtc,
    long RecentTasksCompleted,
    long RecentTokensSeen,
    long RecentWallClockMs,
    double RecentAverageStaleness,
    double RecentAverageLossAfter,
    DateTimeOffset? LastEventUtc,
    long LiveTasksCompleted,
    long LiveTokensSeen,
    double LiveTokensPerSecond,
    double LiveAverageStaleness,
    string? CurrentTaskId,
    DateTimeOffset? CurrentTaskStartedUtc,
    double? SecondsOnCurrentTask);

public sealed class GetDashboardSnapshotQueryHandler : IQueryHandler<GetDashboardSnapshotQuery, DashboardSnapshot>
{
    /// <summary>Time window used for smoothed per-worker averages.</summary>
    public static readonly TimeSpan RecentWindow = TimeSpan.FromMinutes(15);

    /// <summary>Short window driving the progress/ETA and live tok/s columns.</summary>
    public static readonly TimeSpan LiveWindow = TimeSpan.FromMinutes(1);

    private readonly SqliteWorkQueueStore _workQueue;
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly WeightApplicationService _weights;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly TimeProvider _time;

    public GetDashboardSnapshotQueryHandler(
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

    public Task<Result<DashboardSnapshot>> HandleAsync(
        GetDashboardSnapshotQuery query,
        CallContext context)
    {
        var now = _time.GetUtcNow();
        var recentStart = now - RecentWindow;
        var liveStart = now - LiveWindow;

        var staleAfterSeconds = Math.Max(1, _options.CurrentValue.StaleWorkerThresholdSeconds);
        var staleAfter = TimeSpan.FromSeconds(staleAfterSeconds);
        var taskCounts = new TaskCounts(
            Pending:             _workQueue.CountByState(WorkTaskState.Pending),
            Assigned:            _workQueue.CountByState(WorkTaskState.Assigned),
            Done:                _workQueue.CountByState(WorkTaskState.Done),
            Failed:              _workQueue.CountByState(WorkTaskState.Failed),
            SoftExpiredButAlive: _workQueue.CountSoftExpiredButAlive(staleAfter),
            StuckDead:           _workQueue.CountStuckDead(staleAfter));

        var active   = _workerStore.CountByState(WorkerState.Active);
        var draining = _workerStore.CountByState(WorkerState.Draining);
        var gone     = _workerStore.CountByState(WorkerState.Gone);
        // Shared-key model: no pre-provisioned client list. "Configured"
        // now means "ever-registered" — sum of all known worker rows.
        var workerCounts = new WorkerCounts(
            Configured: active + draining + gone,
            Active:     active,
            Draining:   draining,
            Gone:       gone);

        var recentGlobal = _telemetry.AggregateGlobal(recentStart);
        var liveGlobal   = _telemetry.AggregateGlobal(liveStart);
        var recentByWorker = _telemetry.AggregateByWorker(recentStart);
        var liveByWorker   = _telemetry.AggregateByWorker(liveStart);
        var recentByClient = recentByWorker.ToDictionary(entry => entry.ClientId, StringComparer.Ordinal);
        var liveByClient   = liveByWorker.ToDictionary(entry => entry.ClientId, StringComparer.Ordinal);
        var currentByClient = _workQueue.ListAssignedByWorker();

        var liveWindowSeconds = LiveWindow.TotalSeconds;
        var progress = BuildFleetProgress(taskCounts, liveGlobal, liveWindowSeconds, now);

        var rows = _workerStore.ListAll()
            .Select(w =>
            {
                recentByClient.TryGetValue(w.WorkerId, out var recent);
                liveByClient.TryGetValue(w.WorkerId, out var live);
                currentByClient.TryGetValue(w.WorkerId, out var current);

                var liveTokens = live?.TokensSeen ?? 0L;
                var liveTokPerSec = liveWindowSeconds > 0d
                    ? liveTokens / liveWindowSeconds
                    : 0d;

                double? secondsOnTask = null;
                if (current is { AssignedAtUtc: { } assignedAt })
                {
                    secondsOnTask = Math.Max(0d, (now - assignedAt).TotalSeconds);
                }

                return new DashboardWorkerRow(
                    ClientId:               w.WorkerId,
                    Name:                   w.Name,
                    State:                  w.State,
                    CpuThreads:             w.CpuThreads,
                    TokensPerSecond:        w.TokensPerSecond,
                    RecommendedTokensPerTask: w.RecommendedTokensPerTask,
                    RegisteredAtUtc:        w.RegisteredAtUtc,
                    LastHeartbeatUtc:       w.LastHeartbeatUtc,
                    RecentTasksCompleted:   recent?.TasksCompleted ?? 0,
                    RecentTokensSeen:       recent?.TokensSeen ?? 0,
                    RecentWallClockMs:      recent?.WallClockMs ?? 0,
                    RecentAverageStaleness: recent?.AverageStaleness ?? 0d,
                    RecentAverageLossAfter: recent?.AverageLossAfter ?? 0d,
                    LastEventUtc:           recent?.LastEventUtc,
                    LiveTasksCompleted:     live?.TasksCompleted ?? 0,
                    LiveTokensSeen:         liveTokens,
                    LiveTokensPerSecond:    liveTokPerSec,
                    LiveAverageStaleness:   live?.AverageStaleness ?? 0d,
                    CurrentTaskId:          current?.TaskId,
                    CurrentTaskStartedUtc:  current?.AssignedAtUtc,
                    SecondsOnCurrentTask:   secondsOnTask);
            })
            // Sort by most recent activity first: prefer telemetry
            // LastEventUtc (real work) and fall back to heartbeat so
            // workers that have registered but done nothing yet still
            // order against each other.
            .OrderByDescending(r => r.LastEventUtc ?? r.LastHeartbeatUtc)
            .ToList();

        var snapshot = new DashboardSnapshot(
            CurrentWeightVersion:  _weights.CurrentVersion,
            WeightDimension:       _weights.Dimension,
            Tasks:                 taskCounts,
            Workers:               workerCounts,
            RecentGlobalTelemetry: recentGlobal,
            LiveGlobalTelemetry:   liveGlobal,
            Progress:              progress,
            WorkerRows:            rows,
            GeneratedAtUtc:        now);

        return Task.FromResult(Result<DashboardSnapshot>.Success(snapshot));
    }

    private static FleetProgress BuildFleetProgress(
        TaskCounts tasks,
        GlobalTelemetryAggregate live,
        double liveWindowSeconds,
        DateTimeOffset now)
    {
        var total = tasks.Pending + tasks.Assigned + tasks.Done + tasks.Failed;
        var completed = tasks.Done;
        var remaining = tasks.Pending + tasks.Assigned;
        var percent = total > 0 ? (double)completed / total : 0d;

        var tasksPerSec = liveWindowSeconds > 0d
            ? live.TasksCompleted / liveWindowSeconds
            : 0d;
        var tokensPerSec = liveWindowSeconds > 0d
            ? live.TokensSeen / liveWindowSeconds
            : 0d;

        double? etaSeconds = tasksPerSec > 0d && remaining > 0
            ? remaining / tasksPerSec
            : (double?)null;
        DateTimeOffset? etaUtc = etaSeconds.HasValue
            ? now.AddSeconds(etaSeconds.Value)
            : (DateTimeOffset?)null;

        return new FleetProgress(
            TotalTasks:          total,
            CompletedTasks:      completed,
            RemainingTasks:      remaining,
            PercentComplete:     percent,
            TasksPerSecondLive:  tasksPerSec,
            TokensPerSecondLive: tokensPerSec,
            EtaSeconds:          etaSeconds,
            EtaUtc:              etaUtc,
            LiveWindowSeconds:   liveWindowSeconds);
    }
}
