using System;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// BackgroundService that periodically runs two janitor passes:
///
/// <list type="number">
///   <item>
///     <see cref="SqliteWorkerRegistryStore.SweepStaleWorkers"/>
///     transitions <c>Active</c> workers whose last heartbeat is older
///     than <see cref="CoordinatorOptions.StaleWorkerThresholdSeconds"/>
///     to <c>Gone</c> so the <c>/status</c> dashboard and any
///     task-sizing logic stops counting silent workers.
///   </item>
///   <item>
///     <see cref="SqliteWorkQueueStore.RecycleTimedOutAssignments"/>
///     returns assigned tasks whose deadline has passed back to the
///     <c>Pending</c> queue so another worker can pick them up. This is
///     what makes the queue robust against a worker that grabbed a
///     task and then went silent.
///   </item>
/// </list>
///
/// <para>
/// The sweep cadence defaults to one-third of the stale threshold so
/// a worker that misses exactly one heartbeat has time to come back
/// before being swept. Tests can inject a
/// <see cref="TimeProvider"/> to drive the clock deterministically.
/// </para>
/// </summary>
public sealed class StaleSweeperService : BackgroundService
{
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteWorkQueueStore _queueStore;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<StaleSweeperService> _logger;
    private readonly TimeProvider _time;

    public StaleSweeperService(
        SqliteWorkerRegistryStore workerStore,
        SqliteWorkQueueStore queueStore,
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<StaleSweeperService> logger,
        TimeProvider time)
    {
        _workerStore = workerStore;
        _queueStore = queueStore;
        _options = options;
        _logger = logger;
        _time = time;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("StaleSweeperService started.");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                SweepOnce();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Stale sweep iteration failed; will retry on the next tick.");
            }

            var interval = ComputeSweepInterval();
            try
            {
                await Task.Delay(interval, _time, stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        _logger.LogInformation("StaleSweeperService stopped.");
    }

    /// <summary>
    /// Runs a single sweep pass synchronously. Exposed so tests can
    /// drive the sweeper deterministically without spinning the
    /// BackgroundService's ExecuteAsync loop.
    /// </summary>
    public SweepResult SweepOnce()
    {
        var opts = _options.CurrentValue;
        var staleThreshold = TimeSpan.FromSeconds(opts.StaleWorkerThresholdSeconds);

        var workersSwept = _workerStore.SweepStaleWorkers(staleThreshold);
        var tasksRecycled = _queueStore.RecycleTimedOutAssignments();

        if (workersSwept > 0 || tasksRecycled > 0)
        {
            _logger.LogInformation(
                "Sweep iteration transitioned {Workers} stale worker(s) to Gone and recycled {Tasks} timed-out task(s).",
                workersSwept,
                tasksRecycled);
        }

        return new SweepResult(workersSwept, tasksRecycled);
    }

    private TimeSpan ComputeSweepInterval()
    {
        var threshold = _options.CurrentValue.StaleWorkerThresholdSeconds;
        if (threshold <= 0)
        {
            return TimeSpan.FromSeconds(60);
        }

        var third = Math.Max(5, threshold / 3);
        return TimeSpan.FromSeconds(third);
    }
}

/// <summary>
/// Count of transitions a single sweep iteration performed. Returned
/// by <see cref="StaleSweeperService.SweepOnce"/> so tests (and the
/// /status endpoint, eventually) can assert on the outcome.
/// </summary>
public sealed record SweepResult(int WorkersSweptToGone, int TasksRecycled);
