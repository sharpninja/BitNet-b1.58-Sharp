using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the <c>/register</c> endpoint when an
/// authenticated worker sends its initial
/// <see cref="WorkerRegistrationRequest"/>. Carries the bits the
/// handler needs (already-authenticated client id, request payload)
/// and returns the fields the worker expects in response.
/// </summary>
public sealed record RegisterWorkerCommand(
    string ClientId,
    WorkerRegistrationRequest Request) : ICommand<WorkerRegistrationResponse>;

/// <summary>
/// Handler for <see cref="RegisterWorkerCommand"/>. Upserts the
/// worker row, computes the recommended task size from the
/// capability report, and composes the <see cref="WorkerRegistrationResponse"/>
/// using values from <see cref="CoordinatorOptions"/>.
/// </summary>
public sealed class RegisterWorkerCommandHandler : ICommandHandler<RegisterWorkerCommand, WorkerRegistrationResponse>
{
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly TimeProvider _time;
    private readonly ILogger<RegisterWorkerCommandHandler> _logger;

    public RegisterWorkerCommandHandler(
        SqliteWorkerRegistryStore workerStore,
        IOptionsMonitor<CoordinatorOptions> options,
        TimeProvider time,
        ILogger<RegisterWorkerCommandHandler> logger)
    {
        _workerStore = workerStore;
        _options = options;
        _time = time;
        _logger = logger;
    }

    public Task<Result<WorkerRegistrationResponse>> HandleAsync(
        RegisterWorkerCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (command.Request is null)
        {
            return Task.FromResult(Result<WorkerRegistrationResponse>.Failure("Request body is missing."));
        }

        if (string.IsNullOrWhiteSpace(command.ClientId))
        {
            return Task.FromResult(Result<WorkerRegistrationResponse>.Failure("Authenticated client_id is missing."));
        }

        var opts = _options.CurrentValue;
        var recommendedTokens = TaskSizingCalculator.RecommendedTokensPerTask(
            command.Request.Capability.TokensPerSecond,
            TimeSpan.FromSeconds(opts.TargetTaskDurationSeconds),
            opts.FullStepEfficiency);

        var now = _time.GetUtcNow();
        _workerStore.Upsert(new WorkerRecord(
            WorkerId: command.ClientId,
            Name: string.IsNullOrWhiteSpace(command.Request.WorkerName) ? command.ClientId : command.Request.WorkerName,
            CpuThreads: command.Request.Capability.CpuThreads,
            TokensPerSecond: command.Request.Capability.TokensPerSecond,
            RecommendedTokensPerTask: recommendedTokens,
            ProcessArchitecture: command.Request.ProcessArchitecture,
            OsDescription: command.Request.OsDescription,
            RegisteredAtUtc: now,
            LastHeartbeatUtc: now,
            State: WorkerState.Active));

        _logger.LogInformation(
            "Registered worker {ClientId} ({Name}) at {RegisteredAt} with recommended task size {Tokens}.",
            command.ClientId,
            command.Request.WorkerName,
            now,
            recommendedTokens);

        return Task.FromResult(Result<WorkerRegistrationResponse>.Success(
            new WorkerRegistrationResponse(
                WorkerId: command.ClientId,
                BearerToken: string.Empty,
                InitialWeightVersion: opts.InitialWeightVersion,
                RecommendedTokensPerTask: recommendedTokens,
                HeartbeatIntervalSeconds: opts.HeartbeatIntervalSeconds,
                ServerTime: now)));
    }
}
