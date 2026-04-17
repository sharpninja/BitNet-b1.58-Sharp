using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the /admin/tasks "Requeue failed" button.
/// Flips every Failed task back to Pending so a subsequent dequeue
/// cycle can hand it to a healthy worker. Intended for recovery
/// from transient worker-side crashes — the row's original work
/// parameters are preserved.
/// </summary>
public sealed record RequeueFailedTasksCommand() : ICommand<RequeueFailedTasksResult>;

public sealed record RequeueFailedTasksResult(int Requeued);

public sealed class RequeueFailedTasksCommandHandler
    : ICommandHandler<RequeueFailedTasksCommand, RequeueFailedTasksResult>
{
    private readonly SqliteWorkQueueStore _queue;
    private readonly ILogger<RequeueFailedTasksCommandHandler> _logger;

    public RequeueFailedTasksCommandHandler(
        SqliteWorkQueueStore queue,
        ILogger<RequeueFailedTasksCommandHandler> logger)
    {
        _queue = queue;
        _logger = logger;
    }

    public Task<Result<RequeueFailedTasksResult>> HandleAsync(
        RequeueFailedTasksCommand command,
        CallContext context)
    {
        var requeued = _queue.RequeueFailedTasks();
        _logger.LogInformation("Requeued {Count} Failed task(s) to Pending.", requeued);
        return Task.FromResult(Result<RequeueFailedTasksResult>.Success(
            new RequeueFailedTasksResult(requeued)));
    }
}
