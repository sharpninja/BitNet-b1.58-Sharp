using System;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// Internal domain record for a worker row in the coordinator's SQLite
/// <c>workers</c> table. This type is distinct from the wire-format
/// <see cref="BitNetSharp.Distributed.Contracts.WorkerRegistrationRequest"/>
/// and <see cref="BitNetSharp.Distributed.Contracts.WorkerRegistrationResponse"/>
/// so coordinator-only fields — specifically the bearer-token hash that
/// workers never see — never leak into the wire protocol.
///
/// <para>
/// <see cref="BearerTokenHash"/> stores a base64-encoded SHA-256 digest
/// of the bearer token the coordinator issued. The coordinator never
/// persists the raw token itself; the bearer-auth middleware re-hashes
/// the incoming header and looks up the worker by hash so leaking the
/// database file alone would not compromise the fleet.
/// </para>
/// </summary>
public sealed record WorkerRecord(
    string WorkerId,
    string Name,
    string BearerTokenHash,
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
