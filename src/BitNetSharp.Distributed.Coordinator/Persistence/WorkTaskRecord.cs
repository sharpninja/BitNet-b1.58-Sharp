using System;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// Internal domain record for a single training task row in the
/// coordinator's SQLite work queue. This is distinct from the wire-format
/// <see cref="BitNetSharp.Distributed.Contracts.WorkTaskAssignment"/>
/// because the domain record carries coordinator-only metadata
/// (assignee, attempt count, timestamps) that workers never see.
/// </summary>
public sealed record WorkTaskRecord(
    string TaskId,
    long WeightVersion,
    string ShardId,
    long ShardOffset,
    long ShardLength,
    long TokensPerTask,
    int KLocalSteps,
    string HyperparametersJson,
    WorkTaskState State,
    string? AssignedWorkerId,
    DateTimeOffset? AssignedAtUtc,
    DateTimeOffset? DeadlineUtc,
    int Attempt,
    DateTimeOffset CreatedAtUtc,
    DateTimeOffset? CompletedAtUtc);

/// <summary>
/// Lifecycle states for a task in the work queue. The coordinator only
/// ever transitions tasks along a small number of legal edges; invalid
/// transitions throw.
///
///     Pending  →  Assigned  →  Done
///       ↑            |
///       └────────────┴─────→ Failed (permanent)
///
/// A timed-out Assigned task drops back to Pending so another worker
/// can pick it up, and its <c>attempt</c> counter increments.
/// </summary>
public enum WorkTaskState
{
    Pending,
    Assigned,
    Done,
    Failed
}
