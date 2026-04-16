using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the admin dashboard when the operator
/// clicks "Drain" or "Gone" on a worker row. Flips the worker's
/// lifecycle state in the SQLite registry so the task sweeper and
/// the /status dashboard reflect the new desired state.
/// </summary>
public sealed record MarkWorkerStateCommand(
    string WorkerId,
    WorkerState NewState) : ICommand<WorkerStateResult>;

/// <summary>
/// Result of a successful <see cref="MarkWorkerStateCommand"/>.
/// </summary>
public sealed record WorkerStateResult(
    string WorkerId,
    WorkerState NewState);

public sealed class MarkWorkerStateCommandHandler : ICommandHandler<MarkWorkerStateCommand, WorkerStateResult>
{
    public const string UnknownWorkerCode = "unknown_worker";
    public const string IllegalTransitionCode = "illegal_transition";

    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly ILogger<MarkWorkerStateCommandHandler> _logger;

    public MarkWorkerStateCommandHandler(
        SqliteWorkerRegistryStore workerStore,
        ILogger<MarkWorkerStateCommandHandler> logger)
    {
        _workerStore = workerStore;
        _logger = logger;
    }

    public Task<Result<WorkerStateResult>> HandleAsync(
        MarkWorkerStateCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (string.IsNullOrWhiteSpace(command.WorkerId))
        {
            return Task.FromResult(Result<WorkerStateResult>.Failure("WorkerId must not be empty."));
        }

        var success = command.NewState switch
        {
            WorkerState.Draining => _workerStore.MarkDraining(command.WorkerId),
            WorkerState.Gone     => _workerStore.MarkGone(command.WorkerId),
            _ => false
        };

        if (!success)
        {
            if (command.NewState is not (WorkerState.Draining or WorkerState.Gone))
            {
                return Task.FromResult(Result<WorkerStateResult>.Failure(
                    $"{IllegalTransitionCode}: admin dashboard cannot transition workers back to '{command.NewState}'."));
            }

            return Task.FromResult(Result<WorkerStateResult>.Failure(
                $"{UnknownWorkerCode}: no worker registered with id '{command.WorkerId}'."));
        }

        _logger.LogInformation(
            "Admin dashboard transitioned worker {WorkerId} to {NewState}.",
            command.WorkerId,
            command.NewState);

        return Task.FromResult(Result<WorkerStateResult>.Success(
            new WorkerStateResult(command.WorkerId, command.NewState)));
    }
}
