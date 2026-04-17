using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using McpServer.Cqrs;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// State-changing command dispatched by <c>GET /work</c>. Atomically
/// claims the oldest pending task for the authenticated worker and
/// returns the matching wire-format assignment, or a null payload
/// wrapped in <see cref="Result{T}.Success"/> when the queue is
/// empty. Treated as a command rather than a query because the
/// dequeue transitions task state from Pending → Assigned.
/// </summary>
public sealed record ClaimNextTaskCommand(string ClientId) : ICommand<WorkTaskAssignment?>;

/// <summary>
/// Handler that owns the claim transaction and the DTO mapping from
/// <see cref="WorkTaskRecord"/> to
/// <see cref="WorkTaskAssignment"/>.
/// </summary>
public sealed class ClaimNextTaskCommandHandler : ICommandHandler<ClaimNextTaskCommand, WorkTaskAssignment?>
{
    private readonly SqliteWorkQueueStore _workQueue;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly WeightApplicationService _weights;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly TimeProvider _time;

    public ClaimNextTaskCommandHandler(
        SqliteWorkQueueStore workQueue,
        IOptionsMonitor<CoordinatorOptions> options,
        WeightApplicationService weights,
        SqliteTelemetryStore telemetry,
        TimeProvider time)
    {
        _workQueue = workQueue;
        _options = options;
        _weights = weights;
        _telemetry = telemetry;
        _time = time;
    }

    public Task<Result<WorkTaskAssignment?>> HandleAsync(
        ClaimNextTaskCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (string.IsNullOrWhiteSpace(command.ClientId))
        {
            return Task.FromResult(Result<WorkTaskAssignment?>.Failure("Authenticated client_id is missing."));
        }

        var opts = _options.CurrentValue;
        // Prefer measured real-training throughput for lease sizing:
        // calibration-time tok/s is computed from a synthetic forward
        // pass and overestimates real backprop speed by ~2 orders of
        // magnitude, which expires claims before the worker can submit.
        // Fall back to 2× the configured target duration when no prior
        // gradient exists for this worker.
        var measuredTps = _telemetry.GetMeasuredTokensPerSecond(command.ClientId);
        var fallbackSeconds = opts.TargetTaskDurationSeconds * 2;
        TimeSpan LeaseFor(long tokensPerTask)
        {
            const double SafetyMultiplier = 1.5;
            const double HeadroomSeconds = 60.0;
            const double FloorSeconds = 60.0;
            double seconds = fallbackSeconds;
            if (measuredTps is { } tps && tps > 0.0)
            {
                seconds = (tokensPerTask / tps) * SafetyMultiplier + HeadroomSeconds;
            }
            return TimeSpan.FromSeconds(Math.Max(FloorSeconds, seconds));
        }

        var claimed = _workQueue.TryClaimNextPending(command.ClientId, LeaseFor);
        if (claimed is null)
        {
            return Task.FromResult(Result<WorkTaskAssignment?>.Success(null));
        }

        // Claim-time freshness: the task row stores the weight version
        // the operator enqueued it against, but async SGD wants workers
        // to train against the CURRENT global version so staleness stays
        // bounded no matter how long the task sat in the queue. Override
        // the assignment's weight version + URL with whatever the
        // WeightApplicationService holds right now.
        var currentVersion = _weights.CurrentVersion;
        var baseUrl = opts.BaseUrl.TrimEnd('/');
        WorkTaskAssignment? assignment = new WorkTaskAssignment(
            TaskId: claimed.TaskId,
            WeightVersion: currentVersion,
            WeightUrl: $"{baseUrl}/weights/{currentVersion}",
            ShardId: claimed.ShardId,
            ShardOffset: claimed.ShardOffset,
            ShardLength: claimed.ShardLength,
            TokensPerTask: claimed.TokensPerTask,
            KLocalSteps: claimed.KLocalSteps,
            HyperparametersJson: claimed.HyperparametersJson,
            DeadlineUtc: claimed.DeadlineUtc ?? _time.GetUtcNow().Add(LeaseFor(claimed.TokensPerTask)));

        return Task.FromResult(Result<WorkTaskAssignment?>.Success(assignment));
    }
}
