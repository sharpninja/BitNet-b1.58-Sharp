using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the <c>/heartbeat</c> endpoint. Touches
/// the worker's last-heartbeat timestamp and returns the
/// <see cref="HeartbeatResponse"/> the worker echoes back.
/// </summary>
public sealed record SubmitHeartbeatCommand(
    string ClientId,
    HeartbeatRequest Request) : ICommand<HeartbeatResponse>;

/// <summary>
/// Outcome indicator used by the endpoint layer to decide whether
/// the caller should be told to re-register (410 Gone) instead of
/// receiving the normal 200 OK response.
/// </summary>
public sealed class SubmitHeartbeatCommandHandler : ICommandHandler<SubmitHeartbeatCommand, HeartbeatResponse>
{
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly TimeProvider _time;

    /// <summary>
    /// Machine-readable failure code the endpoint layer looks for so
    /// it can respond with HTTP 410 Gone instead of 500. See
    /// <see cref="ICommandHandler{TCommand,TResult}"/> contract —
    /// the handler uses <c>Result.Failure(code)</c> as a lightweight
    /// channel for the "worker must re-register" outcome.
    /// </summary>
    public const string UnregisteredFailureCode = "unregistered";

    public SubmitHeartbeatCommandHandler(
        SqliteWorkerRegistryStore workerStore,
        TimeProvider time)
    {
        _workerStore = workerStore;
        _time = time;
    }

    public Task<Result<HeartbeatResponse>> HandleAsync(
        SubmitHeartbeatCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (command.Request is null)
        {
            return Task.FromResult(Result<HeartbeatResponse>.Failure("Heartbeat body is missing."));
        }

        if (string.IsNullOrWhiteSpace(command.ClientId))
        {
            return Task.FromResult(Result<HeartbeatResponse>.Failure("Authenticated client_id is missing."));
        }

        var touched = _workerStore.TouchHeartbeat(command.ClientId);
        if (!touched)
        {
            return Task.FromResult(Result<HeartbeatResponse>.Failure(UnregisteredFailureCode));
        }

        return Task.FromResult(Result<HeartbeatResponse>.Success(
            new HeartbeatResponse(
                ShouldDrain: false,
                RecommendedTokensPerTaskOverride: null,
                ServerTime: _time.GetUtcNow())));
    }
}
