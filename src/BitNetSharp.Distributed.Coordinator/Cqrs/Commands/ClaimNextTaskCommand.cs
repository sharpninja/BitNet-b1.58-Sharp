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
    private readonly TimeProvider _time;

    public ClaimNextTaskCommandHandler(
        SqliteWorkQueueStore workQueue,
        IOptionsMonitor<CoordinatorOptions> options,
        WeightApplicationService weights,
        TimeProvider time)
    {
        _workQueue = workQueue;
        _options = options;
        _weights = weights;
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
        var leaseDuration = TimeSpan.FromSeconds(opts.TargetTaskDurationSeconds * 2);
        var claimed = _workQueue.TryClaimNextPending(command.ClientId, leaseDuration);
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
            DeadlineUtc: claimed.DeadlineUtc ?? _time.GetUtcNow().Add(leaseDuration));

        return Task.FromResult(Result<WorkTaskAssignment?>.Success(assignment));
    }
}
