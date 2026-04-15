using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the admin /admin/tasks/enqueue endpoint
/// that inserts <see cref="Count"/> pending tasks into the SQLite
/// work queue, all referencing the same corpus shard range but with
/// monotonically-increasing byte offsets. This is how the operator
/// seeds a training run through the browser or a curl script
/// before any workers show up.
/// </summary>
public sealed record EnqueueTasksCommand(
    string ShardId,
    long ShardStartOffset,
    long ShardStride,
    long TokensPerTask,
    int KLocalSteps,
    string HyperparametersJson,
    long WeightVersion,
    int Count) : ICommand<EnqueueTasksResult>;

/// <summary>
/// Outcome of a successful bulk enqueue. Returned so the operator's
/// admin page can show a confirmation banner with the number of
/// tasks actually inserted.
/// </summary>
public sealed record EnqueueTasksResult(int Inserted, string FirstTaskId, string LastTaskId);

public sealed class EnqueueTasksCommandHandler : ICommandHandler<EnqueueTasksCommand, EnqueueTasksResult>
{
    private readonly SqliteWorkQueueStore _workQueue;
    private readonly TimeProvider _time;
    private readonly ILogger<EnqueueTasksCommandHandler> _logger;

    public EnqueueTasksCommandHandler(
        SqliteWorkQueueStore workQueue,
        TimeProvider time,
        ILogger<EnqueueTasksCommandHandler> logger)
    {
        _workQueue = workQueue;
        _time = time;
        _logger = logger;
    }

    public Task<Result<EnqueueTasksResult>> HandleAsync(
        EnqueueTasksCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (command.Count <= 0)
        {
            return Task.FromResult(Result<EnqueueTasksResult>.Failure("Count must be positive."));
        }

        if (string.IsNullOrWhiteSpace(command.ShardId))
        {
            return Task.FromResult(Result<EnqueueTasksResult>.Failure("ShardId must not be empty."));
        }

        if (command.TokensPerTask <= 0)
        {
            return Task.FromResult(Result<EnqueueTasksResult>.Failure("TokensPerTask must be positive."));
        }

        if (command.KLocalSteps <= 0)
        {
            return Task.FromResult(Result<EnqueueTasksResult>.Failure("KLocalSteps must be positive."));
        }

        var now = _time.GetUtcNow();
        var hpJson = string.IsNullOrWhiteSpace(command.HyperparametersJson) ? "{}" : command.HyperparametersJson;
        var stride = command.ShardStride > 0 ? command.ShardStride : command.TokensPerTask;

        var firstTaskId = string.Empty;
        var lastTaskId = string.Empty;
        var inserted = 0;
        for (var i = 0; i < command.Count; i++)
        {
            var taskId = $"task-{Guid.NewGuid():N}";
            var offset = command.ShardStartOffset + (long)i * stride;

            _workQueue.EnqueuePending(new WorkTaskRecord(
                TaskId: taskId,
                WeightVersion: command.WeightVersion,
                ShardId: command.ShardId,
                ShardOffset: offset,
                ShardLength: stride,
                TokensPerTask: command.TokensPerTask,
                KLocalSteps: command.KLocalSteps,
                HyperparametersJson: hpJson,
                State: WorkTaskState.Pending,
                AssignedWorkerId: null,
                AssignedAtUtc: null,
                DeadlineUtc: null,
                Attempt: 0,
                CreatedAtUtc: now,
                CompletedAtUtc: null));

            if (i == 0)
            {
                firstTaskId = taskId;
            }

            lastTaskId = taskId;
            inserted++;
        }

        _logger.LogInformation(
            "Enqueued {Inserted} tasks against shard {ShardId} at weight version {WeightVersion}.",
            inserted,
            command.ShardId,
            command.WeightVersion);

        return Task.FromResult(Result<EnqueueTasksResult>.Success(
            new EnqueueTasksResult(inserted, firstTaskId, lastTaskId)));
    }
}
