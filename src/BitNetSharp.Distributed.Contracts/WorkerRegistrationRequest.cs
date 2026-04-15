using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Payload a worker POSTs to <c>/register</c> when it first comes online.
/// The coordinator responds with a <see cref="WorkerRegistrationResponse"/>
/// that contains the per-worker bearer token and the recommended task
/// sizing for subsequent <c>/work</c> polls.
/// </summary>
/// <param name="WorkerName">Human-readable worker identifier. Typically
/// the container hostname.</param>
/// <param name="EnrollmentKey">Pre-shared secret that authorizes the
/// worker to join the pool. The coordinator compares this against its
/// configured enrollment key and rejects the registration on mismatch.</param>
/// <param name="ProcessArchitecture">ARM64 / X64 / etc. Used for logging
/// and diagnostics only.</param>
/// <param name="OsDescription">Runtime OS description reported by
/// <c>RuntimeInformation.OSDescription</c>. Logging / diagnostics only.</param>
/// <param name="Capability">BenchmarkDotNet calibration result from the
/// worker's startup run.</param>
public sealed record WorkerRegistrationRequest(
    string WorkerName,
    string EnrollmentKey,
    string ProcessArchitecture,
    string OsDescription,
    WorkerCapabilityDto Capability);
