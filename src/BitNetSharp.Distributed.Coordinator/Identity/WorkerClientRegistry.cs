using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Duende.IdentityServer.Models;

namespace BitNetSharp.Distributed.Coordinator.Identity;

/// <summary>
/// Thread-safe in-memory registry of OAuth 2.0 client-credentials
/// clients (one per worker). Seeded from <see cref="CoordinatorOptions.WorkerClients"/>
/// at process startup and mutable at runtime via the admin rotate
/// endpoint, which regenerates a client's secret on demand.
///
/// <para>
/// The registry stores the plain-text secret alongside a hashed copy
/// Duende can use at the token endpoint. Plain-text is required
/// specifically so the admin page can display it for operator copy;
/// in production deployments the admin page must itself be behind an
/// auth wall (the coordinator's basic auth scheme).
/// </para>
/// </summary>
public sealed class WorkerClientRegistry
{
    private readonly ConcurrentDictionary<string, WorkerClientEntry> _clients = new(StringComparer.Ordinal);

    /// <summary>
    /// Seeds the registry with the initial client list parsed from
    /// configuration. Safe to call once from <c>Program.cs</c>.
    /// </summary>
    public void Seed(IEnumerable<WorkerClientOptions> workerClients)
    {
        ArgumentNullException.ThrowIfNull(workerClients);

        foreach (var worker in workerClients)
        {
            if (string.IsNullOrWhiteSpace(worker.ClientId))
            {
                throw new InvalidOperationException("Worker client entry has an empty ClientId.");
            }

            if (string.IsNullOrWhiteSpace(worker.ClientSecret))
            {
                throw new InvalidOperationException($"Worker client '{worker.ClientId}' has an empty ClientSecret.");
            }

            var entry = new WorkerClientEntry(
                ClientId: worker.ClientId,
                PlainTextSecret: worker.ClientSecret,
                DisplayName: string.IsNullOrWhiteSpace(worker.DisplayName) ? worker.ClientId : worker.DisplayName);

            _clients[worker.ClientId] = entry;
        }
    }

    /// <summary>
    /// Returns the full set of currently configured clients. The
    /// snapshot is safe to iterate even if another thread mutates the
    /// registry mid-enumeration.
    /// </summary>
    public IReadOnlyList<WorkerClientEntry> ListAll()
    {
        return _clients.Values
            .OrderBy(entry => entry.ClientId, StringComparer.Ordinal)
            .ToList();
    }

    /// <summary>
    /// Returns the entry for the given client id, or <c>null</c> if no
    /// such client is registered.
    /// </summary>
    public WorkerClientEntry? Find(string clientId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientId);
        return _clients.TryGetValue(clientId, out var entry) ? entry : null;
    }

    /// <summary>
    /// Generates a fresh cryptographically random client secret for
    /// the given client, replaces the in-memory entry, and returns the
    /// new plaintext secret. Throws if no such client exists.
    /// </summary>
    public string Rotate(string clientId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientId);

        if (!_clients.TryGetValue(clientId, out var existing))
        {
            throw new KeyNotFoundException($"Unknown worker client '{clientId}'.");
        }

        var freshSecret = GenerateSecret();
        var replacement = existing with { PlainTextSecret = freshSecret };
        _clients[clientId] = replacement;
        return freshSecret;
    }

    /// <summary>
    /// Adapts the current set of entries into the Duende
    /// <see cref="Client"/> model consumed by the in-memory client
    /// store. Each worker client is configured for the OAuth 2.0
    /// client-credentials grant and the <c>bitnet-worker</c> API scope.
    /// Re-evaluated on every call so consumers always see the latest
    /// registry state (important for admin-rotate support).
    /// </summary>
    public IEnumerable<Client> ToDuendeClients(int accessTokenLifetimeSeconds)
    {
        foreach (var entry in _clients.Values)
        {
            yield return BuildDuendeClient(entry, accessTokenLifetimeSeconds);
        }
    }

    /// <summary>
    /// Builds the Duende <see cref="Client"/> that corresponds to a
    /// single registry entry. Exposed as a static so
    /// <see cref="CompositeClientStore"/> can reuse the exact same
    /// shape when it is asked for a client by id.
    /// </summary>
    public static Client BuildDuendeClient(WorkerClientEntry entry, int accessTokenLifetimeSeconds)
    {
        return new Client
        {
            ClientId = entry.ClientId,
            ClientName = entry.DisplayName,
            AllowedGrantTypes = GrantTypes.ClientCredentials,
            ClientSecrets =
            {
                new Secret(Sha256Base64(entry.PlainTextSecret))
            },
            AllowedScopes = { IdentityServerResources.WorkerScopeName },
            AccessTokenLifetime = accessTokenLifetimeSeconds,
            AccessTokenType = AccessTokenType.Jwt,
            AllowOfflineAccess = false,
            RequireClientSecret = true,
            Claims =
            {
                new ClientClaim("worker_display_name", entry.DisplayName)
            }
        };
    }

    /// <summary>
    /// Returns true if the registry currently contains no workers.
    /// Used by Program.cs to log a warning at startup when the pool
    /// is empty so operators know they forgot to set env vars.
    /// </summary>
    public bool IsEmpty => _clients.IsEmpty;

    /// <summary>
    /// Count of currently registered workers, for /status dashboard.
    /// </summary>
    public int Count => _clients.Count;

    private static string GenerateSecret()
    {
        const int byteLength = 32;
        var bytes = new byte[byteLength];
        RandomNumberGenerator.Fill(bytes);
        // URL-safe base64: no '+', '/', or '=' so operators can paste
        // the secret into env files and shell heredocs without escapes.
        return Convert.ToBase64String(bytes).Replace('+', '-').Replace('/', '_').TrimEnd('=');
    }

    /// <summary>
    /// Duende's <see cref="Secret"/> type expects the secret value to
    /// be pre-hashed. We store the SHA-256 hex digest of the plaintext
    /// secret so the Duende hasher matches on validation. Kept
    /// internal and deliberately simple so we don't take a dep on
    /// IdentityModel just for its <c>ToSha256</c> extension method.
    /// </summary>
    /// <summary>
    /// Computes the SHA-256 hash of the given plaintext string and
    /// returns it as standard base64 — the exact format Duende's
    /// default hashed-shared-secret validator expects. Internal so
    /// the test project can reach it via InternalsVisibleTo if we
    /// ever add assertions over secret hashing.
    /// </summary>
    internal static string Sha256Base64(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        var hash = SHA256.HashData(bytes);
        return Convert.ToBase64String(hash);
    }
}

/// <summary>
/// Immutable snapshot of a single worker client entry in the registry.
/// </summary>
public sealed record WorkerClientEntry(
    string ClientId,
    string PlainTextSecret,
    string DisplayName);
