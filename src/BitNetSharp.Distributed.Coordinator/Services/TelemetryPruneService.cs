using System;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// BackgroundService that periodically prunes old rows from
/// <c>gradient_events</c> and <c>worker_logs</c> so the
/// coordinator's SQLite database does not grow without bound.
/// Runs once per hour by default.
/// </summary>
public sealed class TelemetryPruneService : BackgroundService
{
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<TelemetryPruneService> _logger;
    private readonly string _connectionString;

    public TelemetryPruneService(
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<TelemetryPruneService> logger)
    {
        _options = options;
        _logger = logger;
        var coord = options.CurrentValue;
        _connectionString = $"Data Source={coord.DatabasePath};Cache=Shared";
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("TelemetryPruneService started.");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(TimeSpan.FromHours(1), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            try
            {
                PruneOnce();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Prune iteration failed; will retry in 1 hour.");
            }
        }

        _logger.LogInformation("TelemetryPruneService stopped.");
    }

    /// <summary>
    /// Deletes rows older than the configured retention periods.
    /// Exposed publicly so tests can drive a single pass.
    /// </summary>
    public PruneResult PruneOnce()
    {
        var opts = _options.CurrentValue;
        var telemetryCutoff = DateTimeOffset.UtcNow.AddDays(-Math.Max(1, opts.TelemetryRetentionDays)).ToUnixTimeSeconds();
        var logCutoff = DateTimeOffset.UtcNow.AddDays(-Math.Max(1, opts.LogRetentionDays)).ToUnixTimeSeconds();

        using var connection = new SqliteConnection(_connectionString);
        connection.Open();

        var telemetryDeleted = 0;
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = "DELETE FROM gradient_events WHERE received_at < $cutoff;";
            cmd.Parameters.AddWithValue("$cutoff", telemetryCutoff);
            telemetryDeleted = cmd.ExecuteNonQuery();
        }

        var logsDeleted = 0;
        using (var cmd = connection.CreateCommand())
        {
            cmd.CommandText = "DELETE FROM worker_logs WHERE received_at < $cutoff;";
            cmd.Parameters.AddWithValue("$cutoff", logCutoff);
            logsDeleted = cmd.ExecuteNonQuery();
        }

        if (telemetryDeleted > 0 || logsDeleted > 0)
        {
            _logger.LogInformation(
                "Pruned {TelemetryDeleted} telemetry rows and {LogsDeleted} log rows.",
                telemetryDeleted,
                logsDeleted);
        }

        return new PruneResult(telemetryDeleted, logsDeleted);
    }
}

/// <summary>
/// Counts of rows deleted in a single prune pass.
/// </summary>
public sealed record PruneResult(int TelemetryRowsDeleted, int LogRowsDeleted);
