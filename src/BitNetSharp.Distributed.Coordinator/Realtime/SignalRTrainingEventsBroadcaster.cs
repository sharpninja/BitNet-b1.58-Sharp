using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Distributed.Coordinator.Realtime;

/// <summary>
/// Production <see cref="ITrainingEventsBroadcaster"/>. Thin wrapper
/// over <see cref="IHubContext{T}"/> that swallows transport errors so
/// broadcast failures cannot fail the gradient-accept response. The
/// admin page listens on method names <c>"GradientAccepted"</c> and
/// <c>"SnapshotTick"</c>.
/// </summary>
public sealed class SignalRTrainingEventsBroadcaster : ITrainingEventsBroadcaster
{
    private readonly IHubContext<TrainingEventsHub> _hub;
    private readonly ILogger<SignalRTrainingEventsBroadcaster> _logger;

    public SignalRTrainingEventsBroadcaster(
        IHubContext<TrainingEventsHub> hub,
        ILogger<SignalRTrainingEventsBroadcaster> logger)
    {
        _hub = hub;
        _logger = logger;
    }

    public async Task BroadcastGradientAcceptedAsync(
        GradientAcceptedBroadcast broadcast,
        CancellationToken ct = default)
    {
        try
        {
            await _hub.Clients.All.SendAsync("GradientAccepted", broadcast, ct);
        }
        catch (System.Exception ex)
        {
            _logger.LogWarning(ex,
                "SignalR GradientAccepted broadcast failed for task {TaskId}",
                broadcast.TaskId);
        }
    }

    public async Task BroadcastSnapshotTickAsync(
        SnapshotTickBroadcast broadcast,
        CancellationToken ct = default)
    {
        try
        {
            await _hub.Clients.All.SendAsync("SnapshotTick", broadcast, ct);
        }
        catch (System.Exception ex)
        {
            _logger.LogWarning(ex, "SignalR SnapshotTick broadcast failed");
        }
    }
}
