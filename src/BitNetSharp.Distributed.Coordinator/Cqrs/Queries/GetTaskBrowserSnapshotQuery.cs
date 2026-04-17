using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Queries;

/// <summary>
/// Query backing <c>/admin/task-browser</c>. Pulls the most recent
/// queued-or-active tasks, the most recent finished-or-failed tasks,
/// and resolves the shard ID of each row to an absolute on-disk path
/// so the admin UI can show where the shard lives.
/// </summary>
public sealed class GetTaskBrowserSnapshotQuery : IQuery<TaskBrowserSnapshot>
{
    public int Limit { get; }

    public GetTaskBrowserSnapshotQuery(int limit = 200)
    {
        if (limit <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(limit), limit, "limit must be positive");
        }
        Limit = limit;
    }
}

/// <summary>
/// Bundle of queued + finished task rows and the coordinator's corpus
/// directory (so the UI can surface expected shard paths even when the
/// file is missing).
/// </summary>
public sealed record TaskBrowserSnapshot(
    IReadOnlyList<TaskBrowserRow> Queued,
    IReadOnlyList<TaskBrowserRow> Finished,
    string CorpusDirectory,
    DateTimeOffset GeneratedAtUtc);

/// <summary>
/// Denormalised row the Razor page binds against. Carries everything
/// needed to render one line: task ids, the shard coordinates, the
/// resolved shard path (or <c>null</c> if missing), timestamps, and
/// the current lifecycle state.
/// </summary>
public sealed record TaskBrowserRow(
    string TaskId,
    WorkTaskState State,
    long WeightVersion,
    string ShardId,
    long ShardOffset,
    long ShardLength,
    long TokensPerTask,
    int KLocalSteps,
    string? ResolvedShardPath,
    string ExpectedShardPath,
    bool ShardExists,
    string? AssignedWorkerId,
    DateTimeOffset? AssignedAtUtc,
    DateTimeOffset? DeadlineUtc,
    int Attempt,
    DateTimeOffset CreatedAtUtc,
    DateTimeOffset? CompletedAtUtc);

public sealed class GetTaskBrowserSnapshotQueryHandler
    : IQueryHandler<GetTaskBrowserSnapshotQuery, TaskBrowserSnapshot>
{
    private readonly SqliteWorkQueueStore _workQueue;
    private readonly CoordinatorOptions _options;
    private readonly TimeProvider _time;

    public GetTaskBrowserSnapshotQueryHandler(
        SqliteWorkQueueStore workQueue,
        CoordinatorOptions options,
        TimeProvider? time = null)
    {
        _workQueue = workQueue ?? throw new ArgumentNullException(nameof(workQueue));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _time = time ?? TimeProvider.System;
    }

    public Task<Result<TaskBrowserSnapshot>> HandleAsync(
        GetTaskBrowserSnapshotQuery query,
        CallContext context)
    {
        var queued = _workQueue.ListByStates(
            new[] { WorkTaskState.Pending, WorkTaskState.Assigned },
            query.Limit);
        var finished = _workQueue.ListByStates(
            new[] { WorkTaskState.Done, WorkTaskState.Failed },
            query.Limit);

        var dbPath = _options.DatabasePath;
        var corpusDir = CorpusShardLocator.GetCorpusDirectory(dbPath);

        var snapshot = new TaskBrowserSnapshot(
            Queued: Map(queued, dbPath),
            Finished: Map(finished, dbPath),
            CorpusDirectory: corpusDir,
            GeneratedAtUtc: _time.GetUtcNow());

        return Task.FromResult(Result<TaskBrowserSnapshot>.Success(snapshot));
    }

    private static IReadOnlyList<TaskBrowserRow> Map(IReadOnlyList<WorkTaskRecord> rows, string dbPath)
    {
        var mapped = new List<TaskBrowserRow>(rows.Count);
        foreach (var r in rows)
        {
            var resolved = CorpusShardLocator.TryResolve(dbPath, r.ShardId);
            var expected = CorpusShardLocator.GetExpectedBinPath(dbPath, r.ShardId);
            mapped.Add(new TaskBrowserRow(
                TaskId: r.TaskId,
                State: r.State,
                WeightVersion: r.WeightVersion,
                ShardId: r.ShardId,
                ShardOffset: r.ShardOffset,
                ShardLength: r.ShardLength,
                TokensPerTask: r.TokensPerTask,
                KLocalSteps: r.KLocalSteps,
                ResolvedShardPath: resolved,
                ExpectedShardPath: expected,
                ShardExists: resolved is not null,
                AssignedWorkerId: r.AssignedWorkerId,
                AssignedAtUtc: r.AssignedAtUtc,
                DeadlineUtc: r.DeadlineUtc,
                Attempt: r.Attempt,
                CreatedAtUtc: r.CreatedAtUtc,
                CompletedAtUtc: r.CompletedAtUtc));
        }
        return mapped;
    }
}
