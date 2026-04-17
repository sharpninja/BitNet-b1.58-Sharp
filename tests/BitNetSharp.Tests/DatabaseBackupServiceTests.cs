using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests for <see cref="DatabaseBackupService"/>, the nightly
/// backup loop that snapshots the coordinator SQLite DB + weight
/// store under <c>CoordinatorOptions.BackupRoot</c>.
/// </summary>
public sealed class DatabaseBackupServiceTests : IDisposable
{
    private readonly string _root;
    private readonly string _dbPath;
    private readonly string _weightsDir;
    private readonly string _backupRoot;
    private readonly FakeTimeProvider _time;

    public DatabaseBackupServiceTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "bkpsvc-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
        _dbPath = Path.Combine(_root, "coordinator.db");
        _weightsDir = Path.Combine(_root, "weights");
        _backupRoot = Path.Combine(_root, "backups");
        Directory.CreateDirectory(_weightsDir);
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 17, 2, 0, 0, TimeSpan.Zero));

        SeedDatabase();
        SeedWeights();
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best-effort */ }
    }

    private void SeedDatabase()
    {
        using var conn = new SqliteConnection($"Data Source={_dbPath}");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1),(2),(3);";
        cmd.ExecuteNonQuery();
    }

    private void SeedWeights()
    {
        File.WriteAllBytes(Path.Combine(_weightsDir, "v0000000001.bin"), new byte[] { 1, 2, 3, 4 });
        File.WriteAllText(Path.Combine(_weightsDir, "v0000000001.sha256"), "deadbeef");
    }

    private DatabaseBackupService NewService(int retentionDays = 14)
    {
        var options = new CoordinatorOptions
        {
            DatabasePath = _dbPath,
            BackupRoot = _backupRoot,
            BackupRetentionDays = retentionDays,
            BackupIntervalHours = 24,
        };
        return new DatabaseBackupService(
            new StaticOptionsMonitor<CoordinatorOptions>(options),
            NullLogger<DatabaseBackupService>.Instance,
            new BackupHealth(),
            _time);
    }

    [Fact]
    public async Task RunOnceAsync_creates_timestamped_subdir_with_db_and_weights()
    {
        var svc = NewService();

        var result = await svc.RunOnceAsync();

        Assert.True(Directory.Exists(result.BackupPath), $"backup dir missing: {result.BackupPath}");
        Assert.True(File.Exists(Path.Combine(result.BackupPath, "coordinator.db")));
        Assert.True(File.Exists(Path.Combine(result.BackupPath, "weights", "v0000000001.bin")));
        Assert.True(File.Exists(Path.Combine(result.BackupPath, "weights", "v0000000001.sha256")));
    }

    [Fact]
    public async Task RunOnceAsync_backup_db_passes_integrity_check()
    {
        var svc = NewService();
        var result = await svc.RunOnceAsync();

        var backupDb = Path.Combine(result.BackupPath, "coordinator.db");
        using var conn = new SqliteConnection($"Data Source={backupDb};Mode=ReadOnly");
        conn.Open();
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "PRAGMA integrity_check;";
        var check = (string?)cmd.ExecuteScalar();

        Assert.Equal("ok", check);
    }

    [Fact]
    public async Task RunOnceAsync_records_success_on_backup_health()
    {
        var health = new BackupHealth();
        var options = new CoordinatorOptions
        {
            DatabasePath = _dbPath,
            BackupRoot = _backupRoot,
            BackupRetentionDays = 14,
        };
        var svc = new DatabaseBackupService(
            new StaticOptionsMonitor<CoordinatorOptions>(options),
            NullLogger<DatabaseBackupService>.Instance,
            health,
            _time);

        await svc.RunOnceAsync();

        var snap = health.Snapshot();
        Assert.Equal(0, snap.ConsecutiveFailures);
        Assert.Equal(_time.GetUtcNow(), snap.LastSuccessAtUtc);
        Assert.NotNull(snap.LastBackupPath);
    }

    [Fact]
    public async Task RunOnceAsync_records_failure_on_exception()
    {
        var health = new BackupHealth();
        var options = new CoordinatorOptions
        {
            DatabasePath = Path.Combine(_root, "nonexistent-dir", "missing.db"),
            BackupRoot = _backupRoot,
            BackupRetentionDays = 14,
        };
        var svc = new DatabaseBackupService(
            new StaticOptionsMonitor<CoordinatorOptions>(options),
            NullLogger<DatabaseBackupService>.Instance,
            health,
            _time);

        await Assert.ThrowsAnyAsync<Exception>(() => svc.RunOnceAsync());

        var snap = health.Snapshot();
        Assert.Equal(1, snap.ConsecutiveFailures);
        Assert.NotNull(snap.LastFailureMessage);
    }

    [Fact]
    public async Task Retention_sweep_deletes_subdirs_older_than_threshold()
    {
        Directory.CreateDirectory(_backupRoot);
        var oldDir = Path.Combine(_backupRoot, "20260301T020000");
        var freshDir = Path.Combine(_backupRoot, "20260416T020000");
        Directory.CreateDirectory(oldDir);
        Directory.CreateDirectory(freshDir);
        File.WriteAllText(Path.Combine(oldDir, "coordinator.db"), "old");
        File.WriteAllText(Path.Combine(freshDir, "coordinator.db"), "fresh");

        var svc = NewService(retentionDays: 14);

        await svc.RunOnceAsync();

        Assert.False(Directory.Exists(oldDir), "old backup dir should have been pruned");
        Assert.True(Directory.Exists(freshDir), "fresh backup dir should survive");
    }

    [Fact]
    public async Task RunOnceAsync_backup_subdir_name_is_utc_timestamp()
    {
        var svc = NewService();

        var result = await svc.RunOnceAsync();

        var leaf = Path.GetFileName(result.BackupPath);
        // yyyyMMdd'T'HHmmss — 15 chars, 'T' at index 8
        Assert.Equal(15, leaf.Length);
        Assert.Equal('T', leaf[8]);
        Assert.Equal("20260417T020000", leaf);
    }
}
