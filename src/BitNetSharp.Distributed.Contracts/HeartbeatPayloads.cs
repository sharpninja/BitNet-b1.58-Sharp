using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Worker → coordinator heartbeat. Sent on the cadence specified in the
/// initial <see cref="WorkerRegistrationResponse"/>. Includes the
/// worker's current status so the coordinator's dashboard can
/// distinguish idle workers from workers mid-task.
/// </summary>
/// <param name="WorkerId">Worker identifier assigned at registration.</param>
/// <param name="Status">Current worker state: "idle", "training",
/// "draining", "benchmarking".</param>
/// <param name="CurrentTaskId">Task ID the worker is currently
/// executing, if any.</param>
/// <param name="TokensSeenSinceLastHeartbeat">Running counter of tokens
/// processed since the previous heartbeat. Used by the coordinator for
/// aggregate throughput telemetry.</param>
public sealed record HeartbeatRequest(
    string WorkerId,
    string Status,
    string? CurrentTaskId,
    long TokensSeenSinceLastHeartbeat);

/// <summary>
/// Coordinator → worker heartbeat response. Carries any out-of-band
/// commands the coordinator wants the worker to honor, such as an order
/// to start draining before shutdown.
/// </summary>
/// <param name="ShouldDrain">When true, the worker must finish its
/// current task (if any), POST a final heartbeat, and exit cleanly.</param>
/// <param name="RecommendedTokensPerTaskOverride">If the coordinator has
/// recomputed the per-worker task size (for example after observing
/// real training-step efficiency), this carries the new value.
/// <c>null</c> means keep the previously recommended size.</param>
/// <param name="ServerTime">Coordinator clock at response time.</param>
public sealed record HeartbeatResponse(
    bool ShouldDrain,
    long? RecommendedTokensPerTaskOverride,
    DateTimeOffset ServerTime);
