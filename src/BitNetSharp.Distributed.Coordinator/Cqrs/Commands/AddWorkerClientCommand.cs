using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Identity;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command that registers a brand-new worker OAuth client with the
/// coordinator at runtime. Avoids the service-restart dance that
/// used to be required for each new worker machine: operators can
/// hit <c>POST /admin/clients</c> with a client id (and optional
/// display name), get back the freshly generated secret, and use
/// it immediately in an install script.
/// </summary>
public sealed record AddWorkerClientCommand(
    string ClientId,
    string? DisplayName) : ICommand<AddWorkerClientResult>;

/// <summary>
/// Result of a successful <see cref="AddWorkerClientCommand"/>
/// dispatch. The plaintext secret is returned once, at creation
/// time — reading it later requires <see cref="WorkerClientRegistry"/>
/// access on the /admin/api-keys page.
/// </summary>
public sealed record AddWorkerClientResult(
    string ClientId,
    string DisplayName,
    string ClientSecret);

/// <summary>
/// Handler for <see cref="AddWorkerClientCommand"/>. Delegates to
/// <see cref="WorkerClientRegistry.Add"/> and surfaces "already
/// exists" conflicts as <see cref="Result{T}.Failure"/> so the
/// endpoint can translate them to HTTP 409 instead of throwing.
/// </summary>
public sealed class AddWorkerClientCommandHandler
    : ICommandHandler<AddWorkerClientCommand, AddWorkerClientResult>
{
    private readonly WorkerClientRegistry _registry;
    private readonly ILogger<AddWorkerClientCommandHandler> _logger;

    public AddWorkerClientCommandHandler(
        WorkerClientRegistry registry,
        ILogger<AddWorkerClientCommandHandler> logger)
    {
        _registry = registry;
        _logger = logger;
    }

    public Task<Result<AddWorkerClientResult>> HandleAsync(
        AddWorkerClientCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (string.IsNullOrWhiteSpace(command.ClientId))
        {
            return Task.FromResult(Result<AddWorkerClientResult>.Failure("ClientId must not be empty."));
        }

        WorkerClientEntry entry;
        try
        {
            entry = _registry.Add(command.ClientId, command.DisplayName);
        }
        catch (InvalidOperationException ex)
        {
            return Task.FromResult(Result<AddWorkerClientResult>.Failure(ex.Message));
        }

        _logger.LogInformation(
            "Admin registered new worker client {ClientId} ({DisplayName}) via CQRS dispatch.",
            entry.ClientId,
            entry.DisplayName);

        return Task.FromResult(Result<AddWorkerClientResult>.Success(
            new AddWorkerClientResult(entry.ClientId, entry.DisplayName, entry.PlainTextSecret)));
    }
}
