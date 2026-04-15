using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Coordinator response to a successful <see cref="WorkerRegistrationRequest"/>.
/// Carries the per-worker bearer token the worker must attach to every
/// subsequent request, the initial weight version pointer so the worker
/// can download the starting snapshot, and the recommended task size the
/// coordinator derived from the capability report.
/// </summary>
/// <param name="WorkerId">Opaque unique identifier assigned by the
/// coordinator. Echoed back on every subsequent request.</param>
/// <param name="BearerToken">Secret the worker must send in the
/// <c>Authorization: Bearer</c> header on every subsequent request.</param>
/// <param name="InitialWeightVersion">Version number of the current
/// coordinator weights. Workers download
/// <c>/weights/{InitialWeightVersion}</c> before accepting their first
/// task.</param>
/// <param name="RecommendedTokensPerTask">Number of training tokens the
/// coordinator will put in each task for this worker, based on its
/// capability report and the ten-minute target budget.</param>
/// <param name="HeartbeatIntervalSeconds">How often the worker should
/// POST a heartbeat. The coordinator cancels assigned tasks whose worker
/// stops heartbeating.</param>
/// <param name="ServerTime">Coordinator's current UTC clock at response
/// time. Workers log drift but do not rely on it for correctness.</param>
public sealed record WorkerRegistrationResponse(
    string WorkerId,
    string BearerToken,
    long InitialWeightVersion,
    long RecommendedTokensPerTask,
    int HeartbeatIntervalSeconds,
    DateTimeOffset ServerTime);
