using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the <c>/gradient</c> endpoint when a
/// worker reports a completed task. Validates worker ownership and
/// transitions the task state from <c>Assigned</c> to <c>Done</c>.
/// Gradient decoding + global weight apply are deferred to the
/// Phase D-4 commit.
/// </summary>
public sealed record SubmitGradientCommand(
    string ClientId,
    GradientSubmission Submission) : ICommand<GradientAcceptance>;

/// <summary>
/// Lightweight value object returned by the gradient handler on
/// the happy path. The endpoint layer serializes this straight to
/// JSON as the HTTP response body.
/// </summary>
public sealed record GradientAcceptance(
    string TaskId,
    string WorkerId,
    long TokensSeen,
    bool Accepted);

/// <summary>
/// Handler for <see cref="SubmitGradientCommand"/>. Uses
/// <c>Result.Failure</c> with sentinel codes so the endpoint layer
/// can branch to the right HTTP status without re-validating.
/// </summary>
public sealed class SubmitGradientCommandHandler : ICommandHandler<SubmitGradientCommand, GradientAcceptance>
{
    private readonly SqliteWorkQueueStore _workQueue;
    private readonly ILogger<SubmitGradientCommandHandler> _logger;

    /// <summary>Returned when the submission's worker_id does not match the JWT.</summary>
    public const string WorkerMismatchCode = "worker_mismatch";

    /// <summary>Returned when the task is not currently assigned to this worker.</summary>
    public const string TaskNotAssignedCode = "task_not_assigned";

    public SubmitGradientCommandHandler(
        SqliteWorkQueueStore workQueue,
        ILogger<SubmitGradientCommandHandler> logger)
    {
        _workQueue = workQueue;
        _logger = logger;
    }

    public Task<Result<GradientAcceptance>> HandleAsync(
        SubmitGradientCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (command.Submission is null)
        {
            return Task.FromResult(Result<GradientAcceptance>.Failure("Gradient body is missing."));
        }

        if (string.IsNullOrWhiteSpace(command.ClientId)
            || command.ClientId != command.Submission.WorkerId)
        {
            return Task.FromResult(Result<GradientAcceptance>.Failure(WorkerMismatchCode));
        }

        var completed = _workQueue.MarkCompleted(command.Submission.TaskId, command.ClientId);
        if (!completed)
        {
            return Task.FromResult(Result<GradientAcceptance>.Failure(TaskNotAssignedCode));
        }

        _logger.LogInformation(
            "Accepted gradient for task {TaskId} from worker {ClientId}: format={Format}, bytes={Size}, tokens={Tokens}, loss={Loss}",
            command.Submission.TaskId,
            command.ClientId,
            command.Submission.GradientFormat,
            command.Submission.GradientPayload?.Length ?? 0,
            command.Submission.TokensSeen,
            command.Submission.LossAfter);

        return Task.FromResult(Result<GradientAcceptance>.Success(
            new GradientAcceptance(
                TaskId: command.Submission.TaskId,
                WorkerId: command.ClientId,
                TokensSeen: command.Submission.TokensSeen,
                Accepted: true)));
    }
}
