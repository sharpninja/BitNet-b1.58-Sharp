using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Identity;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Queries;

/// <summary>
/// Query that asks the coordinator for the full list of currently
/// configured worker OAuth clients plus their revocation state.
/// Returned by a handler that composes the in-memory
/// <see cref="WorkerClientRegistry"/> with the SQLite
/// <see cref="SqliteClientRevocationStore"/> so the Blazor admin
/// page does not need to know about either source.
/// </summary>
public sealed class GetWorkerClientsQuery : IQuery<IReadOnlyList<WorkerClientView>>
{
}

/// <summary>
/// Projection of a worker client row for the admin UI. Contains the
/// fields the <c>/admin/api-keys</c> page renders as columns.
/// </summary>
public sealed record WorkerClientView(
    string ClientId,
    string DisplayName,
    string ClientSecret,
    DateTimeOffset? RevokedAtUtc);

/// <summary>
/// Handler that materializes every <see cref="WorkerClientEntry"/>
/// in the registry into a <see cref="WorkerClientView"/> and pairs
/// it with the current revocation timestamp from the SQLite store.
/// </summary>
public sealed class GetWorkerClientsQueryHandler : IQueryHandler<GetWorkerClientsQuery, IReadOnlyList<WorkerClientView>>
{
    private readonly WorkerClientRegistry _registry;
    private readonly SqliteClientRevocationStore _revocations;

    public GetWorkerClientsQueryHandler(
        WorkerClientRegistry registry,
        SqliteClientRevocationStore revocations)
    {
        _registry = registry;
        _revocations = revocations;
    }

    public Task<Result<IReadOnlyList<WorkerClientView>>> HandleAsync(
        GetWorkerClientsQuery query,
        CallContext context)
    {
        IReadOnlyList<WorkerClientView> materialized = _registry.ListAll()
            .Select(entry => new WorkerClientView(
                ClientId: entry.ClientId,
                DisplayName: entry.DisplayName,
                ClientSecret: entry.PlainTextSecret,
                RevokedAtUtc: _revocations.GetRevokedAt(entry.ClientId)))
            .ToList();

        return Task.FromResult(Result<IReadOnlyList<WorkerClientView>>.Success(materialized));
    }
}
