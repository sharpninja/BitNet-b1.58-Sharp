using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Duende.IdentityServer.Models;
using Duende.IdentityServer.Stores;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Identity;

/// <summary>
/// Duende <see cref="IClientStore"/> that resolves <see cref="Client"/>
/// lookups against two sources:
///
/// <list type="number">
///   <item>
///     The mutable <see cref="WorkerClientRegistry"/> — each entry is
///     rebuilt into a fresh Duende <see cref="Client"/> on every call
///     so admin rotations take effect immediately even for tokens
///     issued via /connect/token.
///   </item>
///   <item>
///     The single static admin UI client built on demand from
///     <see cref="IdentityServerResources.BuildAdminUiClient"/>
///     using the current <see cref="CoordinatorOptions.BaseUrl"/>.
///   </item>
/// </list>
///
/// <para>
/// Using a custom IClientStore is the recommended path in Duende
/// when the client set can change at runtime; the built-in
/// <c>AddInMemoryClients</c> captures its list at registration time
/// and would miss admin-rotate secret bumps.
/// </para>
/// </summary>
public sealed class CompositeClientStore : IClientStore
{
    private readonly WorkerClientRegistry _registry;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;

    public CompositeClientStore(
        WorkerClientRegistry registry,
        IOptionsMonitor<CoordinatorOptions> options)
    {
        _registry = registry;
        _options = options;
    }

    public Task<Client?> FindClientByIdAsync(string clientId)
    {
        if (string.IsNullOrWhiteSpace(clientId))
        {
            return Task.FromResult<Client?>(null);
        }

        if (clientId == IdentityServerResources.AdminUiClientId)
        {
            var coord = _options.CurrentValue;
            return Task.FromResult<Client?>(
                IdentityServerResources.BuildAdminUiClient(coord.BaseUrl));
        }

        var entry = _registry.Find(clientId);
        if (entry is null)
        {
            return Task.FromResult<Client?>(null);
        }

        var lifetime = _options.CurrentValue.AccessTokenLifetimeSeconds;
        return Task.FromResult<Client?>(
            WorkerClientRegistry.BuildDuendeClient(entry, lifetime));
    }
}
