using System;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Realtime;

/// <summary>
/// Hosted service that ticks every <see cref="TickInterval"/> and
/// broadcasts a lightweight fleet snapshot so the admin
/// <c>/admin/training-status</c> page header keeps updating even when
/// no gradients are flowing. Pulls only the cheap aggregates
/// (current weight version, active worker count, recent tokens/sec);
/// the heavier per-worker rollup is computed on-demand by the query
/// handler.
/// </summary>
public sealed class SnapshotBroadcaster : BackgroundService
{
    /// <summary>Fixed 2-second cadence — matches the existing dashboard refresh.</summary>
    public static readonly TimeSpan TickInterval = TimeSpan.FromSeconds(2);

    /// <summary>Window the fleet tok/s is averaged over.</summary>
    private static readonly TimeSpan LiveWindow = TimeSpan.FromSeconds(60);

    private readonly ITrainingEventsBroadcaster _broadcaster;
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly WeightApplicationService _weights;
    private readonly TimeProvider _time;
    private readonly ILogger<SnapshotBroadcaster> _logger;

    public SnapshotBroadcaster(
        ITrainingEventsBroadcaster broadcaster,
        SqliteWorkerRegistryStore workerStore,
        SqliteTelemetryStore telemetry,
        WeightApplicationService weights,
        TimeProvider time,
        ILogger<SnapshotBroadcaster> logger)
    {
        _broadcaster = broadcaster;
        _workerStore = workerStore;
        _telemetry = telemetry;
        _weights = weights;
        _time = time;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(TickInterval);
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var tick = BuildTick();
                await _broadcaster.BroadcastSnapshotTickAsync(tick, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                return;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "SnapshotBroadcaster tick failed");
            }

            try
            {
                if (!await timer.WaitForNextTickAsync(stoppingToken))
                {
                    return;
                }
            }
            catch (OperationCanceledException)
            {
                return;
            }
        }
    }

    internal SnapshotTickBroadcast BuildTick()
    {
        var now = _time.GetUtcNow();
        var window = _telemetry.AggregateGlobal(now - LiveWindow);
        var tokensPerSec = LiveWindow.TotalSeconds > 0d
            ? window.TokensSeen / LiveWindow.TotalSeconds
            : 0d;
        var activeWorkers = _workerStore.CountByState(WorkerState.Active);

        return new SnapshotTickBroadcast(
            CurrentWeightVersion: _weights.CurrentVersion,
            ActiveWorkerCount: activeWorkers,
            GlobalTokensPerSecond: tokensPerSec,
            FleetEtaSeconds: null,
            GeneratedAtUtc: now);
    }
}
