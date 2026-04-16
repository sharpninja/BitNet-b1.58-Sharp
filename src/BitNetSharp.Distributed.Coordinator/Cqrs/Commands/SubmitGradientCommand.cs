using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the <c>/gradient</c> endpoint when a
/// worker reports a completed task. Phase D-4: decodes the int8+
/// error-feedback gradient payload, applies it to the global
/// weight vector via <see cref="WeightApplicationService"/>, and
/// marks the task <c>Done</c>. Workers whose submissions are
/// rejected (wrong shape, too stale) get a machine-readable
/// failure code the endpoint layer maps to the right HTTP status.
/// </summary>
public sealed record SubmitGradientCommand(
    string ClientId,
    GradientSubmission Submission) : ICommand<GradientAcceptance>;

/// <summary>
/// Value object returned on the happy path. Carries the version
/// the apply produced so the worker can refresh its local weights
/// to that version on its next task.
/// </summary>
public sealed record GradientAcceptance(
    string TaskId,
    string WorkerId,
    long TokensSeen,
    long NewWeightVersion,
    long Staleness,
    float EffectiveLearningRate,
    bool Accepted);

public sealed class SubmitGradientCommandHandler : ICommandHandler<SubmitGradientCommand, GradientAcceptance>
{
    private readonly SqliteWorkQueueStore _workQueue;
    private readonly WeightApplicationService _weights;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly ILogger<SubmitGradientCommandHandler> _logger;

    /// <summary>Returned when the submission's worker_id does not match the JWT.</summary>
    public const string WorkerMismatchCode = "worker_mismatch";

    /// <summary>Returned when the task is not currently assigned to this worker.</summary>
    public const string TaskNotAssignedCode = "task_not_assigned";

    /// <summary>Returned when the payload cannot be decoded.</summary>
    public const string InvalidPayloadCode = "invalid_payload";

    /// <summary>Returned when the gradient shape mismatches the global weight vector.</summary>
    public const string GradientShapeCode = "gradient_shape";

    /// <summary>Returned when the gradient is too stale to apply.</summary>
    public const string StaleGradientCode = "stale_gradient";

    public SubmitGradientCommandHandler(
        SqliteWorkQueueStore workQueue,
        WeightApplicationService weights,
        SqliteTelemetryStore telemetry,
        ILogger<SubmitGradientCommandHandler> logger)
    {
        _workQueue = workQueue;
        _weights = weights;
        _telemetry = telemetry;
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

        // Accept the legacy stub-noop format for backward compatibility
        // with the D-1 smoke tests and the D-2 two-machine proof runs;
        // skip decode + apply in that case.
        var payload = command.Submission.GradientPayload ?? Array.Empty<byte>();
        var isStubNoop = string.Equals(command.Submission.GradientFormat, "stub-noop", StringComparison.OrdinalIgnoreCase);

        long newVersion;
        long staleness;
        float effectiveLr;

        if (isStubNoop)
        {
            newVersion = _weights.CurrentVersion;
            staleness = 0;
            effectiveLr = 0f;
        }
        else if (string.Equals(command.Submission.GradientFormat, Int8GradientCodec.FormatId, StringComparison.OrdinalIgnoreCase))
        {
            if (!Int8GradientCodec.TryDecode(payload, out var gradient, out var decodeError))
            {
                return Task.FromResult(Result<GradientAcceptance>.Failure(
                    $"{InvalidPayloadCode}: {decodeError}"));
            }

            var apply = _weights.Apply(command.Submission.BaseWeightVersion, gradient);
            if (!apply.Accepted)
            {
                var code = apply.Staleness > 0 ? StaleGradientCode : GradientShapeCode;
                _logger.LogInformation(
                    "Rejected gradient from worker {ClientId} for task {TaskId}: {Reason}",
                    command.ClientId,
                    command.Submission.TaskId,
                    apply.Reason);
                return Task.FromResult(Result<GradientAcceptance>.Failure(
                    $"{code}: {apply.Reason}"));
            }

            newVersion = apply.NewVersion;
            staleness = apply.Staleness;
            effectiveLr = apply.EffectiveLearningRate;
        }
        else
        {
            return Task.FromResult(Result<GradientAcceptance>.Failure(
                $"{InvalidPayloadCode}: Unknown gradient format '{command.Submission.GradientFormat}'."));
        }

        var completed = _workQueue.MarkCompleted(command.Submission.TaskId, command.ClientId);
        if (!completed)
        {
            return Task.FromResult(Result<GradientAcceptance>.Failure(TaskNotAssignedCode));
        }

        _telemetry.RecordAccepted(
            clientId: command.ClientId,
            taskId: command.Submission.TaskId,
            tokensSeen: command.Submission.TokensSeen,
            wallClockMs: command.Submission.WallClockMs,
            staleness: staleness,
            effectiveLr: effectiveLr,
            newVersion: newVersion,
            lossAfter: command.Submission.LossAfter);

        _logger.LogInformation(
            "Accepted gradient for task {TaskId} from worker {ClientId}: format={Format}, bytes={Size}, tokens={Tokens}, loss={Loss}, staleness={Staleness}, new_version={NewVersion}, effective_lr={EffectiveLr:F4}",
            command.Submission.TaskId,
            command.ClientId,
            command.Submission.GradientFormat,
            payload.Length,
            command.Submission.TokensSeen,
            command.Submission.LossAfter,
            staleness,
            newVersion,
            effectiveLr);

        return Task.FromResult(Result<GradientAcceptance>.Success(
            new GradientAcceptance(
                TaskId: command.Submission.TaskId,
                WorkerId: command.ClientId,
                TokensSeen: command.Submission.TokensSeen,
                NewWeightVersion: newVersion,
                Staleness: staleness,
                EffectiveLearningRate: effectiveLr,
                Accepted: true)));
    }
}
