using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Identity;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command that rotates the OAuth client secret for an existing
/// worker client and revokes any JWT already issued for that client
/// so the next request from the worker fails until it is restarted
/// with the new secret.
/// </summary>
public sealed record RotateClientSecretCommand(string ClientId) : ICommand<RotationResult>;

/// <summary>
/// Result of a successful <see cref="RotateClientSecretCommand"/>
/// dispatch. The admin UI uses this both to echo the new plaintext
/// secret back to the operator and to show the revoked_at timestamp
/// on the <c>/admin/api-keys</c> page.
/// </summary>
public sealed record RotationResult(
    string ClientId,
    string NewSecret,
    DateTimeOffset RevokedAtUtc);

/// <summary>
/// Handler that drives the two-step rotation: generate a fresh
/// secret in the <see cref="WorkerClientRegistry"/> and stamp the
/// <see cref="SqliteClientRevocationStore"/> with "now". Errors are
/// returned as <see cref="Result{T}.Failure"/> rather than thrown so
/// the admin endpoint can respond with a clean 404 when the caller
/// asks to rotate an unknown client.
/// </summary>
public sealed class RotateClientSecretCommandHandler : ICommandHandler<RotateClientSecretCommand, RotationResult>
{
    private readonly WorkerClientRegistry _registry;
    private readonly SqliteClientRevocationStore _revocations;
    private readonly ILogger<RotateClientSecretCommandHandler> _logger;

    public RotateClientSecretCommandHandler(
        WorkerClientRegistry registry,
        SqliteClientRevocationStore revocations,
        ILogger<RotateClientSecretCommandHandler> logger)
    {
        _registry = registry;
        _revocations = revocations;
        _logger = logger;
    }

    public Task<Result<RotationResult>> HandleAsync(
        RotateClientSecretCommand command,
        CallContext context)
    {
        ArgumentNullException.ThrowIfNull(command);
        if (string.IsNullOrWhiteSpace(command.ClientId))
        {
            return Task.FromResult(Result<RotationResult>.Failure("ClientId must not be empty."));
        }

        string freshSecret;
        try
        {
            freshSecret = _registry.Rotate(command.ClientId);
        }
        catch (KeyNotFoundException)
        {
            return Task.FromResult(
                Result<RotationResult>.Failure($"Client '{command.ClientId}' is not registered."));
        }

        var revokedAt = _revocations.Revoke(command.ClientId);
        _logger.LogWarning(
            "Admin rotated client secret for {ClientId} at {RevokedAt} via CQRS dispatch. Existing JWTs for this client are now invalid.",
            command.ClientId,
            revokedAt);

        return Task.FromResult(Result<RotationResult>.Success(
            new RotationResult(command.ClientId, freshSecret, revokedAt)));
    }
}
