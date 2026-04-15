using System;
using System.Collections.Generic;
using Microsoft.Data.Sqlite;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// SQLite-backed table mapping <c>client_id</c> to the UTC timestamp at
/// which the operator last rotated or revoked that client's credential.
/// The JWT auth middleware consults this store on every authenticated
/// request and rejects tokens whose <c>iat</c> (issued-at) claim is
/// older than the client's <c>revoked_at</c> — this is how the
/// coordinator makes API-key rotation take effect immediately rather
/// than waiting for the natural JWT expiry window.
///
/// <para>
/// The rotation flow is:
///   1. Operator hits <c>POST /admin/rotate/{client_id}</c>.
///   2. Coordinator generates a fresh <c>client_secret</c>, updates
///      the Duende in-memory client store, and calls
///      <see cref="Revoke"/> with "now".
///   3. Every JWT that any worker already holds for that client has an
///      <c>iat</c> older than the new <c>revoked_at</c>, so the first
///      follow-up request from that worker fails with 401.
///   4. The worker operator copies the new secret into the worker's
///      environment and restarts the worker container. The worker
///      re-runs <c>/connect/token</c> with the new secret and obtains
///      a JWT whose <c>iat</c> is newer than the revocation timestamp,
///      which passes the middleware check.
/// </para>
/// </summary>
public sealed class SqliteClientRevocationStore : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly TimeProvider _time;
    private readonly object _writeGate = new();

    public SqliteClientRevocationStore(string connectionString, TimeProvider? time = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        _time = time ?? TimeProvider.System;
        _connection = new SqliteConnection(connectionString);
        _connection.Open();

        ExecuteNonQuery("PRAGMA journal_mode = WAL;");
        ExecuteNonQuery("PRAGMA synchronous = NORMAL;");
        ExecuteNonQuery("PRAGMA busy_timeout = 5000;");
        MigrateSchema();
    }

    private void MigrateSchema()
    {
        ExecuteNonQuery(@"
CREATE TABLE IF NOT EXISTS client_revocations (
    client_id   TEXT PRIMARY KEY,
    revoked_at  INTEGER NOT NULL
);
");
    }

    /// <summary>
    /// Marks the given client's credential as revoked as of the
    /// injected <see cref="TimeProvider"/>'s current time. Safe to call
    /// repeatedly; each call bumps the timestamp forward so any newly
    /// issued JWT with a later <c>iat</c> continues to pass.
    /// </summary>
    public DateTimeOffset Revoke(string clientId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientId);

        var now = _time.GetUtcNow();
        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
INSERT INTO client_revocations (client_id, revoked_at)
VALUES ($client_id, $revoked_at)
ON CONFLICT(client_id) DO UPDATE SET revoked_at = excluded.revoked_at;";
            cmd.Parameters.AddWithValue("$client_id", clientId);
            cmd.Parameters.AddWithValue("$revoked_at", now.ToUnixTimeSeconds());
            cmd.ExecuteNonQuery();
        }

        return now;
    }

    /// <summary>
    /// Returns the UTC timestamp at which the given client's credential
    /// was most recently revoked, or <c>null</c> if it has never been
    /// revoked.
    /// </summary>
    public DateTimeOffset? GetRevokedAt(string clientId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientId);

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT revoked_at FROM client_revocations WHERE client_id = $client_id;";
        cmd.Parameters.AddWithValue("$client_id", clientId);
        var result = cmd.ExecuteScalar();
        if (result is null or DBNull)
        {
            return null;
        }

        var unixSeconds = Convert.ToInt64(result, System.Globalization.CultureInfo.InvariantCulture);
        return DateTimeOffset.FromUnixTimeSeconds(unixSeconds);
    }

    /// <summary>
    /// True when a JWT with the given <c>iat</c> should be rejected for
    /// this client. The comparison uses strict less-than so a token
    /// issued in the same second as the revocation still fails — this
    /// closes a small race window where a worker in the middle of a
    /// rotate call could hold a "fresh" token that actually predates
    /// the operator's intent.
    /// </summary>
    public bool IsIssuedBeforeRevocation(string clientId, DateTimeOffset jwtIssuedAt)
    {
        var revokedAt = GetRevokedAt(clientId);
        if (revokedAt is null)
        {
            return false;
        }

        return jwtIssuedAt <= revokedAt.Value;
    }

    /// <summary>
    /// Lists every client that currently has a revocation timestamp.
    /// Useful for the admin page and for operator scripts.
    /// </summary>
    public IReadOnlyList<(string ClientId, DateTimeOffset RevokedAt)> ListAll()
    {
        var results = new List<(string, DateTimeOffset)>();
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT client_id, revoked_at FROM client_revocations ORDER BY client_id;";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            results.Add((reader.GetString(0), DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(1))));
        }

        return results;
    }

    private void ExecuteNonQuery(string sql)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    public void Dispose()
    {
        _connection.Dispose();
    }
}
