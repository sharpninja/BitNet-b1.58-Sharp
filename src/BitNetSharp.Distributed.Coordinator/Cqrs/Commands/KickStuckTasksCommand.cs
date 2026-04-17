using System;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the dashboard "Kick stuck" button on the
/// StuckDead card. Releases every Assigned task whose deadline has
/// passed AND whose owning worker is missing or has a stale
/// heartbeat, returning those rows to Pending. Mirrors the
/// <c>CountStuckDead</c> predicate so the button clears exactly the
/// rows the card surfaces.
/// </summary>
public sealed record KickStuckTasksCommand() : ICommand<KickStuckTasksResult>;

public sealed record KickStuckTasksResult(int Kicked);

public sealed class KickStuckTasksCommandHandler
    : ICommandHandler<KickStuckTasksCommand, KickStuckTasksResult>
{
    private readonly SqliteWorkQueueStore _queue;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<KickStuckTasksCommandHandler> _logger;

    public KickStuckTasksCommandHandler(
        SqliteWorkQueueStore queue,
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<KickStuckTasksCommandHandler> logger)
    {
        _queue = queue;
        _options = options;
        _logger = logger;
    }

    public Task<Result<KickStuckTasksResult>> HandleAsync(
        KickStuckTasksCommand command,
        CallContext context)
    {
        var staleAfter = TimeSpan.FromSeconds(_options.CurrentValue.StaleWorkerThresholdSeconds);
        var kicked = _queue.KickStuckTasks(staleAfter);
        _logger.LogInformation("Kicked {Count} stuck-dead task(s) back to Pending.", kicked);
        return Task.FromResult(Result<KickStuckTasksResult>.Success(
            new KickStuckTasksResult(kicked)));
    }
}
