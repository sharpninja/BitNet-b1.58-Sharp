using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// A unit of work dispatched by the coordinator to a worker in response
/// to <c>GET /work</c>. Each task tells the worker which weight version
/// to train against, which corpus shard to use, how many local SGD steps
/// to run, and when the task must finish (deadline) so the coordinator
/// can recycle assignments that time out.
/// </summary>
/// <param name="TaskId">Coordinator-assigned unique identifier. The
/// worker echoes this on the matching <c>POST /gradient</c>.</param>
/// <param name="WeightVersion">The integer weight version the worker
/// should base its gradient computation on. If the worker's local copy
/// is stale it must re-download <c>/weights/{WeightVersion}</c> before
/// executing the task.</param>
/// <param name="WeightUrl">Absolute URL the worker uses to fetch the
/// weight blob if it does not already hold that version.</param>
/// <param name="ShardId">Logical identifier of the corpus shard the
/// worker should train on. Paired with <see cref="ShardOffset"/> and
/// <see cref="ShardLength"/>.</param>
/// <param name="ShardOffset">Byte or record offset into the shard at
/// which this task starts.</param>
/// <param name="ShardLength">Byte or record length of the slice
/// assigned to this task. Along with the tokens-per-task value, this is
/// what makes tasks for a slow worker smaller than tasks for a fast
/// worker.</param>
/// <param name="TokensPerTask">Token budget for this task. The worker
/// should stop consuming the shard after it has processed this many
/// tokens even if the byte range has not been exhausted.</param>
/// <param name="KLocalSteps">Number of local SGD steps the worker should
/// run between gradient submissions. K=1 is pure async; larger K
/// amortizes network round trips at the cost of staleness.</param>
/// <param name="HyperparametersJson">Opaque JSON blob of training
/// hyperparameters (learning rate, momentum, etc.) the coordinator wants
/// the worker to apply for this task. Treated as a black box by the
/// Contracts layer so hyperparameter schema evolution does not force a
/// wire-format change.</param>
/// <param name="DeadlineUtc">Wall-clock time after which the coordinator
/// will consider the task abandoned and reassign it.</param>
public sealed record WorkTaskAssignment(
    string TaskId,
    long WeightVersion,
    string WeightUrl,
    string ShardId,
    long ShardOffset,
    long ShardLength,
    long TokensPerTask,
    int KLocalSteps,
    string HyperparametersJson,
    DateTimeOffset DeadlineUtc);
