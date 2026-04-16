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
/// Called on every page load so the operator sees current state;
/// a D-5 follow-up upgrade will swap the page to Blazor
/// interactive-server render for real-time streaming.
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
    IReadOnlyList<DashboardWorkerRow> WorkerRows,
    DateTimeOffset GeneratedAtUtc);

public sealed record TaskCounts(int Pending, int Assigned, int Done, int Failed);

public sealed record WorkerCounts(int Configured, int Active, int Draining, int Gone);

/// <summary>
/// One row on the dashboard's worker table. Combines the static
/// registry info (id, name, cpu threads) with the recent telemetry
/// rollup for that worker.
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
    DateTimeOffset? LastEventUtc);

public sealed class GetDashboardSnapshotQueryHandler : IQueryHandler<GetDashboardSnapshotQuery, DashboardSnapshot>
{
    /// <summary>Time window used by the dashboard's rollup aggregates.</summary>
    public static readonly TimeSpan RecentWindow = TimeSpan.FromMinutes(15);

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
        var windowStart = now - RecentWindow;

        var taskCounts = new TaskCounts(
            Pending:  _workQueue.CountByState(WorkTaskState.Pending),
            Assigned: _workQueue.CountByState(WorkTaskState.Assigned),
            Done:     _workQueue.CountByState(WorkTaskState.Done),
            Failed:   _workQueue.CountByState(WorkTaskState.Failed));

        var workerCounts = new WorkerCounts(
            Configured: _options.CurrentValue.WorkerClients.Count,
            Active:     _workerStore.CountByState(WorkerState.Active),
            Draining:   _workerStore.CountByState(WorkerState.Draining),
            Gone:       _workerStore.CountByState(WorkerState.Gone));

        var globalTelemetry = _telemetry.AggregateGlobal(windowStart);
        var perWorker = _telemetry.AggregateByWorker(windowStart);
        var byClient = perWorker.ToDictionary(entry => entry.ClientId, StringComparer.Ordinal);

        var rows = _workerStore.ListAll()
            .Select(w =>
            {
                byClient.TryGetValue(w.WorkerId, out var telemetry);
                return new DashboardWorkerRow(
                    ClientId:               w.WorkerId,
                    Name:                   w.Name,
                    State:                  w.State,
                    CpuThreads:             w.CpuThreads,
                    TokensPerSecond:        w.TokensPerSecond,
                    RecommendedTokensPerTask: w.RecommendedTokensPerTask,
                    RegisteredAtUtc:        w.RegisteredAtUtc,
                    LastHeartbeatUtc:       w.LastHeartbeatUtc,
                    RecentTasksCompleted:   telemetry?.TasksCompleted ?? 0,
                    RecentTokensSeen:       telemetry?.TokensSeen ?? 0,
                    RecentWallClockMs:      telemetry?.WallClockMs ?? 0,
                    RecentAverageStaleness: telemetry?.AverageStaleness ?? 0d,
                    RecentAverageLossAfter: telemetry?.AverageLossAfter ?? 0d,
                    LastEventUtc:           telemetry?.LastEventUtc);
            })
            .ToList();

        var snapshot = new DashboardSnapshot(
            CurrentWeightVersion: _weights.CurrentVersion,
            WeightDimension:      _weights.Dimension,
            Tasks:                taskCounts,
            Workers:              workerCounts,
            RecentGlobalTelemetry: globalTelemetry,
            WorkerRows:           rows,
            GeneratedAtUtc:       now);

        return Task.FromResult(Result<DashboardSnapshot>.Success(snapshot));
    }
}
