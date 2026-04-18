using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;

namespace BitNetSharp.Distributed.Coordinator.Realtime;

/// <summary>
/// SignalR hub that pushes training activity to the admin
/// <c>/admin/training-status</c> page without polling. Gated by the
/// same <c>AdminPolicy</c> the admin pages use so non-admin cookies
/// cannot subscribe. Two broadcast methods the page listens for:
/// <c>"GradientAccepted"</c> carries per-submission detail, and
/// <c>"SnapshotTick"</c> carries a fleet-wide 2-second heartbeat so
/// idle workers still get live weight-version / ETA updates.
/// </summary>
[Authorize(Policy = "AdminPolicy")]
public sealed class TrainingEventsHub : Hub
{
}

/// <summary>
/// Payload broadcast once per accepted gradient submission.
/// <see cref="ShardId"/> is the <c>tasks.shard_id</c> the task
/// belonged to; it is empty when the task row was pruned before
/// the broadcast fired.
/// </summary>
public sealed record GradientAcceptedBroadcast(
    DateTimeOffset ReceivedAtUtc,
    string ClientId,
    string TaskId,
    string ShardId,
    long TokensSeen,
    long WallClockMs,
    double? MeasuredTokensPerSecond,
    long NewVersion,
    double LossAfter);

/// <summary>
/// Payload broadcast by <see cref="SnapshotBroadcaster"/> on a fixed
/// cadence so the page's header numbers keep ticking even when no
/// gradients are flowing.
/// </summary>
public sealed record SnapshotTickBroadcast(
    long CurrentWeightVersion,
    int ActiveWorkerCount,
    double GlobalTokensPerSecond,
    double? FleetEtaSeconds,
    DateTimeOffset GeneratedAtUtc);

/// <summary>
/// Transport abstraction the gradient command handler calls into to
/// fire a <see cref="GradientAcceptedBroadcast"/>. Separates SignalR
/// from the handler so tests can substitute a capturing fake without
/// spinning up a hub context, and so broadcast failures cannot fail
/// the gradient-accept response.
/// </summary>
public interface ITrainingEventsBroadcaster
{
    Task BroadcastGradientAcceptedAsync(
        GradientAcceptedBroadcast broadcast,
        CancellationToken ct = default);

    Task BroadcastSnapshotTickAsync(
        SnapshotTickBroadcast broadcast,
        CancellationToken ct = default);
}

/// <summary>
/// No-op implementation used when SignalR is not wired (tests that
/// don't care about the push surface). Production registers
/// <c>SignalRTrainingEventsBroadcaster</c> instead.
/// </summary>
public sealed class NoOpTrainingEventsBroadcaster : ITrainingEventsBroadcaster
{
    public Task BroadcastGradientAcceptedAsync(
        GradientAcceptedBroadcast broadcast,
        CancellationToken ct = default) => Task.CompletedTask;

    public Task BroadcastSnapshotTickAsync(
        SnapshotTickBroadcast broadcast,
        CancellationToken ct = default) => Task.CompletedTask;
}
