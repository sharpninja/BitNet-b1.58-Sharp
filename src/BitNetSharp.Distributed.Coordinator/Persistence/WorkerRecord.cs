using System;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// Internal domain record for a worker row in the coordinator's SQLite
/// <c>workers</c> table. The primary key doubles as the Duende
/// IdentityServer <c>client_id</c> that the worker authenticates with
/// on every request — one worker = one OAuth client.
///
/// <para>
/// No bearer-token hash is persisted on this row. Authentication
/// credentials live on the Duende client configuration
/// (which itself is supplied via environment variables at coordinator
/// startup). Validation of the JWT access token is handled by the
/// Microsoft.AspNetCore.Authentication.JwtBearer middleware and does
/// not require a per-request DB lookup.
/// </para>
/// </summary>
public sealed record WorkerRecord(
    string WorkerId,
    string Name,
    int CpuThreads,
    double TokensPerSecond,
    long RecommendedTokensPerTask,
    string? ProcessArchitecture,
    string? OsDescription,
    DateTimeOffset RegisteredAtUtc,
    DateTimeOffset LastHeartbeatUtc,
    WorkerState State);

/// <summary>
/// Lifecycle states for a worker in the coordinator's registry.
///
///     Active   → healthy, eligible for task assignment
///     Draining → finishing in-flight task, not eligible for new work
///     Gone     → heartbeat missed past threshold OR explicit deregistration
/// </summary>
public enum WorkerState
{
    Active,
    Draining,
    Gone
}
