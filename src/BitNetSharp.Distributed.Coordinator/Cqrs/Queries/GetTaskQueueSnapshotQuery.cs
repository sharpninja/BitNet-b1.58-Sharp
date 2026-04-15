using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Queries;

/// <summary>
/// Query that returns a per-state snapshot of the coordinator's
/// work queue for the /admin/tasks dashboard and /status JSON
/// endpoint.
/// </summary>
public sealed class GetTaskQueueSnapshotQuery : IQuery<TaskQueueSnapshot>
{
}

/// <summary>
/// Counts of tasks in each lifecycle state. Four integers rather
/// than a dictionary so the Razor page can bind field-by-field.
/// </summary>
public sealed record TaskQueueSnapshot(
    int Pending,
    int Assigned,
    int Done,
    int Failed);

public sealed class GetTaskQueueSnapshotQueryHandler : IQueryHandler<GetTaskQueueSnapshotQuery, TaskQueueSnapshot>
{
    private readonly SqliteWorkQueueStore _workQueue;

    public GetTaskQueueSnapshotQueryHandler(SqliteWorkQueueStore workQueue)
    {
        _workQueue = workQueue;
    }

    public Task<Result<TaskQueueSnapshot>> HandleAsync(
        GetTaskQueueSnapshotQuery query,
        CallContext context)
    {
        var snapshot = new TaskQueueSnapshot(
            Pending: _workQueue.CountByState(WorkTaskState.Pending),
            Assigned: _workQueue.CountByState(WorkTaskState.Assigned),
            Done: _workQueue.CountByState(WorkTaskState.Done),
            Failed: _workQueue.CountByState(WorkTaskState.Failed));

        return Task.FromResult(Result<TaskQueueSnapshot>.Success(snapshot));
    }
}
