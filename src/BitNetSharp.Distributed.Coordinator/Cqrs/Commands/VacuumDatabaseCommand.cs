using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using McpServer.Cqrs;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Cqrs.Commands;

/// <summary>
/// Command dispatched by the dashboard "Vacuum database" button.
/// Runs <c>PRAGMA wal_checkpoint(TRUNCATE);</c> then <c>VACUUM;</c>
/// against the coordinator's SQLite file to compact free pages and
/// truncate the WAL. In-place only — nightly snapshots are produced
/// separately by
/// <see cref="BitNetSharp.Distributed.Coordinator.Services.DatabaseBackupService"/>.
/// </summary>
public sealed record VacuumDatabaseCommand() : ICommand<VacuumDatabaseResult>;

public sealed record VacuumDatabaseResult(long SizeBeforeBytes, long SizeAfterBytes);

public sealed class VacuumDatabaseCommandHandler
    : ICommandHandler<VacuumDatabaseCommand, VacuumDatabaseResult>
{
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<VacuumDatabaseCommandHandler> _logger;

    public VacuumDatabaseCommandHandler(
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<VacuumDatabaseCommandHandler> logger)
    {
        _options = options;
        _logger = logger;
    }

    public async Task<Result<VacuumDatabaseResult>> HandleAsync(
        VacuumDatabaseCommand command,
        CallContext context)
    {
        var dbPath = _options.CurrentValue.DatabasePath;
        var before = new System.IO.FileInfo(dbPath).Exists
            ? new System.IO.FileInfo(dbPath).Length
            : 0L;

        using (var conn = new SqliteConnection($"Data Source={dbPath}"))
        {
            await conn.OpenAsync().ConfigureAwait(false);
            using (var checkpoint = conn.CreateCommand())
            {
                checkpoint.CommandText = "PRAGMA wal_checkpoint(TRUNCATE);";
                await checkpoint.ExecuteNonQueryAsync().ConfigureAwait(false);
            }
            using (var vacuum = conn.CreateCommand())
            {
                vacuum.CommandText = "VACUUM;";
                await vacuum.ExecuteNonQueryAsync().ConfigureAwait(false);
            }
        }

        var after = new System.IO.FileInfo(dbPath).Exists
            ? new System.IO.FileInfo(dbPath).Length
            : 0L;

        _logger.LogInformation(
            "Vacuumed coordinator DB: {Before} -> {After} bytes.",
            before,
            after);

        return Result<VacuumDatabaseResult>.Success(new VacuumDatabaseResult(before, after));
    }
}
