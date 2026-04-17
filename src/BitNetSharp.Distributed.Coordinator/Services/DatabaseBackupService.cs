using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// BackgroundService that snapshots the coordinator SQLite database +
/// the weight-store directory to
/// <c>CoordinatorOptions.BackupRoot</c> once per
/// <c>BackupIntervalHours</c>. Uses <c>VACUUM INTO</c> for the DB
/// (WAL-safe online copy) and a simple file copy for the immutable
/// weight blobs. Old backup subdirs beyond
/// <c>BackupRetentionDays</c> are pruned on each iteration.
/// </summary>
public sealed class DatabaseBackupService : BackgroundService
{
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<DatabaseBackupService> _logger;
    private readonly BackupHealth _health;
    private readonly TimeProvider _time;

    public DatabaseBackupService(
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<DatabaseBackupService> logger,
        BackupHealth health,
        TimeProvider time)
    {
        _options = options;
        _logger = logger;
        _health = health;
        _time = time;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("DatabaseBackupService started.");

        while (!stoppingToken.IsCancellationRequested)
        {
            var intervalHours = Math.Max(1, _options.CurrentValue.BackupIntervalHours);
            try
            {
                await Task.Delay(TimeSpan.FromHours(intervalHours), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            try
            {
                await RunOnceAsync(stoppingToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Backup iteration failed; will retry next interval.");
            }
        }

        _logger.LogInformation("DatabaseBackupService stopped.");
    }

    /// <summary>
    /// Performs a single backup pass. Exposed publicly so tests (and
    /// a future admin verb) can drive one iteration without waiting
    /// on the hourly timer.
    /// </summary>
    public async Task<BackupResult> RunOnceAsync(CancellationToken ct = default)
    {
        var opts = _options.CurrentValue;
        var now = _time.GetUtcNow();
        var backupRoot = ResolveBackupRoot(opts);
        var stampedDir = Path.Combine(backupRoot, now.UtcDateTime.ToString("yyyyMMdd'T'HHmmss"));

        try
        {
            Directory.CreateDirectory(stampedDir);

            var bytes = await BackupDatabaseAsync(opts.DatabasePath, stampedDir, ct).ConfigureAwait(false);
            bytes += CopyWeights(opts.DatabasePath, stampedDir);

            PruneOldBackups(backupRoot, now, Math.Max(1, opts.BackupRetentionDays));

            _health.RecordSuccess(now, stampedDir);
            _logger.LogInformation(
                "Backup written to {Path} ({Bytes} bytes).",
                stampedDir,
                bytes);

            return new BackupResult(stampedDir, bytes);
        }
        catch (Exception ex)
        {
            _health.RecordFailure(ex, now);
            throw;
        }
    }

    private static string ResolveBackupRoot(CoordinatorOptions opts)
    {
        if (!string.IsNullOrWhiteSpace(opts.BackupRoot)) return opts.BackupRoot;
        var dbDir = Path.GetDirectoryName(Path.GetFullPath(opts.DatabasePath)) ?? ".";
        return Path.Combine(dbDir, "backups");
    }

    private static async Task<long> BackupDatabaseAsync(string dbPath, string destDir, CancellationToken ct)
    {
        var destDb = Path.Combine(destDir, Path.GetFileName(dbPath));
        var source = new SqliteConnection($"Data Source={dbPath};Mode=ReadWriteCreate");
        await source.OpenAsync(ct).ConfigureAwait(false);
        await using (source)
        {
            using var cmd = source.CreateCommand();
            // VACUUM INTO produces a fully self-contained copy that is
            // safe to take while writers hold the source DB — SQLite
            // snapshots pages under a read transaction it manages
            // internally. Simpler than the streaming BackupDatabase API.
            cmd.CommandText = "VACUUM INTO $path;";
            cmd.Parameters.AddWithValue("$path", destDb);
            await cmd.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
        }
        return new FileInfo(destDb).Length;
    }

    private static long CopyWeights(string dbPath, string destDir)
    {
        var weightsDir = Path.Combine(
            Path.GetDirectoryName(Path.GetFullPath(dbPath)) ?? ".",
            "weights");
        if (!Directory.Exists(weightsDir)) return 0;

        var destWeights = Path.Combine(destDir, "weights");
        Directory.CreateDirectory(destWeights);

        long copied = 0;
        foreach (var src in Directory.EnumerateFiles(weightsDir))
        {
            var dst = Path.Combine(destWeights, Path.GetFileName(src));
            File.Copy(src, dst, overwrite: true);
            copied += new FileInfo(dst).Length;
        }
        return copied;
    }

    private static void PruneOldBackups(string backupRoot, DateTimeOffset now, int retentionDays)
    {
        if (!Directory.Exists(backupRoot)) return;
        var cutoff = now.UtcDateTime.AddDays(-retentionDays);

        foreach (var dir in Directory.EnumerateDirectories(backupRoot))
        {
            var leaf = Path.GetFileName(dir);
            if (!DateTime.TryParseExact(
                    leaf,
                    "yyyyMMdd'T'HHmmss",
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Globalization.DateTimeStyles.AssumeUniversal | System.Globalization.DateTimeStyles.AdjustToUniversal,
                    out var stamp))
            {
                continue;
            }
            if (stamp < cutoff)
            {
                try { Directory.Delete(dir, recursive: true); } catch { /* best-effort */ }
            }
        }
    }
}

/// <summary>
/// Outcome of a single backup pass. <see cref="BackupPath"/> is the
/// directory under <c>BackupRoot</c> that holds this snapshot's
/// files.
/// </summary>
public sealed record BackupResult(string BackupPath, long BytesCopied);
