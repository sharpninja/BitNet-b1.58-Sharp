using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.Data.Sqlite;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// SQLite-backed ring buffer of per-gradient telemetry events. Every
/// successful <c>/gradient</c> submission records a single row with
/// the worker id, task id, wall-clock duration, tokens processed,
/// staleness, effective learning rate, and a timestamp. The
/// dashboard query rolls these rows up into per-worker and global
/// statistics; a nightly prune will be wired in D-5.
///
/// <para>
/// The store opens its own WAL-mode connection against the
/// coordinator database file so it can coexist with the other
/// SQLite stores and concurrent writers serialize naturally.
/// </para>
/// </summary>
public sealed class SqliteTelemetryStore : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly TimeProvider _time;
    private readonly object _writeGate = new();

    public SqliteTelemetryStore(string connectionString, TimeProvider? time = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        _time = time ?? TimeProvider.System;
        _connection = new SqliteConnection(connectionString);
        _connection.Open();

        ExecuteNonQuery("PRAGMA journal_mode = WAL;");
        ExecuteNonQuery("PRAGMA synchronous = NORMAL;");
        ExecuteNonQuery("PRAGMA busy_timeout = 5000;");

        MigrateSchema();
    }

    private void MigrateSchema()
    {
        ExecuteNonQuery(@"
CREATE TABLE IF NOT EXISTS gradient_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at     INTEGER NOT NULL,
    client_id       TEXT    NOT NULL,
    task_id         TEXT    NOT NULL,
    tokens_seen     INTEGER NOT NULL,
    wall_clock_ms   INTEGER NOT NULL,
    staleness       INTEGER NOT NULL,
    effective_lr    REAL    NOT NULL,
    new_version     INTEGER NOT NULL,
    loss_after      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_gradient_events_client
    ON gradient_events(client_id, received_at);

CREATE INDEX IF NOT EXISTS ix_gradient_events_received
    ON gradient_events(received_at);
");

        // v2 migration: add nullable measured_tps column so workers that
        // populate GradientSubmission.MeasuredTokensPerSecond can have
        // their authoritative rate stored alongside the derived one.
        // Older rows stay NULL and fall back to derivation in rollups.
        AddColumnIfMissing("gradient_events", "measured_tps", "REAL");
    }

    private void AddColumnIfMissing(string table, string column, string type)
    {
        using var probe = _connection.CreateCommand();
        probe.CommandText = $"PRAGMA table_info({table});";
        using var reader = probe.ExecuteReader();
        while (reader.Read())
        {
            if (string.Equals(reader.GetString(1), column, StringComparison.Ordinal))
            {
                return;
            }
        }
        reader.Close();
        ExecuteNonQuery($"ALTER TABLE {table} ADD COLUMN {column} {type};");
    }

    /// <summary>
    /// Records a single accepted gradient event. Safe to call from
    /// multiple threads.
    /// </summary>
    public void RecordAccepted(
        string clientId,
        string taskId,
        long tokensSeen,
        long wallClockMs,
        long staleness,
        float effectiveLr,
        long newVersion,
        double lossAfter,
        double? measuredTokensPerSecond = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientId);
        ArgumentException.ThrowIfNullOrWhiteSpace(taskId);

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
INSERT INTO gradient_events (
    received_at, client_id, task_id, tokens_seen, wall_clock_ms,
    staleness, effective_lr, new_version, loss_after, measured_tps
) VALUES (
    $received_at, $client_id, $task_id, $tokens_seen, $wall_clock_ms,
    $staleness, $effective_lr, $new_version, $loss_after, $measured_tps
);";
            cmd.Parameters.AddWithValue("$received_at", _time.GetUtcNow().ToUnixTimeSeconds());
            cmd.Parameters.AddWithValue("$client_id", clientId);
            cmd.Parameters.AddWithValue("$task_id", taskId);
            cmd.Parameters.AddWithValue("$tokens_seen", tokensSeen);
            cmd.Parameters.AddWithValue("$wall_clock_ms", wallClockMs);
            cmd.Parameters.AddWithValue("$staleness", staleness);
            cmd.Parameters.AddWithValue("$effective_lr", effectiveLr);
            cmd.Parameters.AddWithValue("$new_version", newVersion);
            cmd.Parameters.AddWithValue("$loss_after", lossAfter);
            cmd.Parameters.AddWithValue("$measured_tps", (object?)measuredTokensPerSecond ?? DBNull.Value);
            cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Returns a per-worker rollup of gradient submissions received
    /// since <paramref name="since"/>. Orders rows by tasks completed
    /// descending so the dashboard's top row is the hottest worker.
    /// </summary>
    public IReadOnlyList<WorkerTelemetryAggregate> AggregateByWorker(DateTimeOffset since)
    {
        var results = new List<WorkerTelemetryAggregate>();
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT client_id,
       COUNT(1),
       COALESCE(SUM(tokens_seen), 0),
       COALESCE(SUM(wall_clock_ms), 0),
       COALESCE(AVG(staleness), 0),
       COALESCE(AVG(loss_after), 0),
       MAX(received_at)
FROM gradient_events
WHERE received_at >= $since
GROUP BY client_id
ORDER BY COUNT(1) DESC;";
        cmd.Parameters.AddWithValue("$since", since.ToUnixTimeSeconds());

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var lastSeenUnix = reader.IsDBNull(6) ? 0L : reader.GetInt64(6);
            results.Add(new WorkerTelemetryAggregate(
                ClientId: reader.GetString(0),
                TasksCompleted: reader.GetInt64(1),
                TokensSeen: reader.GetInt64(2),
                WallClockMs: reader.GetInt64(3),
                AverageStaleness: reader.GetDouble(4),
                AverageLossAfter: reader.GetDouble(5),
                LastEventUtc: lastSeenUnix == 0L ? null : DateTimeOffset.FromUnixTimeSeconds(lastSeenUnix)));
        }

        return results;
    }

    /// <summary>
    /// Returns a single global rollup of all gradient submissions
    /// received since <paramref name="since"/>. Used by the
    /// dashboard's "last N minutes" header panel.
    /// </summary>
    public GlobalTelemetryAggregate AggregateGlobal(DateTimeOffset since)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT COUNT(1),
       COALESCE(SUM(tokens_seen), 0),
       COALESCE(SUM(wall_clock_ms), 0),
       COALESCE(AVG(staleness), 0),
       COALESCE(AVG(loss_after), 0)
FROM gradient_events
WHERE received_at >= $since;";
        cmd.Parameters.AddWithValue("$since", since.ToUnixTimeSeconds());

        using var reader = cmd.ExecuteReader();
        reader.Read();
        return new GlobalTelemetryAggregate(
            TasksCompleted: reader.GetInt64(0),
            TokensSeen: reader.GetInt64(1),
            WallClockMs: reader.GetInt64(2),
            AverageStaleness: reader.GetDouble(3),
            AverageLossAfter: reader.GetDouble(4));
    }

    /// <summary>
    /// Returns the measured real-world tokens/second for a given worker
    /// based on the most recent <paramref name="lookback"/> accepted
    /// gradients inside <paramref name="maxAge"/> (default 30 min).
    /// Derives throughput from stored <c>tokens_seen</c> and
    /// <c>wall_clock_ms</c>, which unlike the calibration-time
    /// <c>workers.tokens_per_sec</c> column reflects real backprop
    /// cost. Returns <c>null</c> when no prior gradients exist inside
    /// the window or when total wall-clock is zero.
    ///
    /// <para>
    /// The time window matters: without it, legacy fast synthetic
    /// gradients (wall_clock_ms in the single-digit-millisecond range)
    /// skew the rollup to 700k+ tok/s, which then produces 60-second
    /// leases that expire before 10-minute real-training tasks finish.
    /// Bounding to 30 minutes means a regime change (e.g. synthetic ⇒
    /// real-corpus workload) naturally decays within the window.
    /// </para>
    /// </summary>
    public double? GetMeasuredTokensPerSecond(
        string clientId,
        int lookback = 8,
        TimeSpan? maxAge = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(clientId);
        if (lookback <= 0) { lookback = 1; }
        var window = maxAge ?? TimeSpan.FromMinutes(30);
        var cutoff = _time.GetUtcNow().Subtract(window).ToUnixTimeSeconds();

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT COALESCE(SUM(tokens_seen), 0), COALESCE(SUM(wall_clock_ms), 0)
FROM (
    SELECT tokens_seen, wall_clock_ms
    FROM gradient_events
    WHERE client_id = $client_id
      AND received_at >= $cutoff
    ORDER BY id DESC
    LIMIT $limit
);";
        cmd.Parameters.AddWithValue("$client_id", clientId);
        cmd.Parameters.AddWithValue("$limit", lookback);
        cmd.Parameters.AddWithValue("$cutoff", cutoff);

        using var reader = cmd.ExecuteReader();
        if (!reader.Read()) return null;
        var tokens = reader.GetInt64(0);
        var wallMs = reader.GetInt64(1);
        if (tokens <= 0 || wallMs <= 0) return null;
        return tokens / (wallMs / 1000.0);
    }

    /// <summary>
    /// Returns the fleet-wide measured tokens/sec over accepted
    /// gradients inside <paramref name="maxAge"/> (default 30 min).
    /// Used by the seed-size feedback loop in <c>seed-real-tasks</c>
    /// to pick <c>tokensPerTask</c> so a newly-seeded task fits the
    /// 10-minute target window on the slowest current worker.
    /// Returns <c>null</c> when no gradients exist in the window or
    /// total wall-clock is zero.
    /// </summary>
    public double? GetGlobalMeasuredTokensPerSecond(TimeSpan? maxAge = null)
    {
        var window = maxAge ?? TimeSpan.FromMinutes(30);
        var cutoff = _time.GetUtcNow().Subtract(window).ToUnixTimeSeconds();

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT COALESCE(SUM(tokens_seen), 0), COALESCE(SUM(wall_clock_ms), 0)
FROM gradient_events
WHERE received_at >= $cutoff;";
        cmd.Parameters.AddWithValue("$cutoff", cutoff);

        using var reader = cmd.ExecuteReader();
        if (!reader.Read()) return null;
        var tokens = reader.GetInt64(0);
        var wallMs = reader.GetInt64(1);
        if (tokens <= 0 || wallMs <= 0) return null;
        return tokens / (wallMs / 1000.0);
    }

    /// <summary>
    /// Total row count in the telemetry table; handy for sanity
    /// checks in tests.
    /// </summary>
    public long TotalCount()
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT COUNT(1) FROM gradient_events;";
        var obj = cmd.ExecuteScalar();
        return obj is null or DBNull ? 0L : Convert.ToInt64(obj, CultureInfo.InvariantCulture);
    }

    private void ExecuteNonQuery(string sql)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    public void Dispose()
    {
        _connection.Dispose();
    }
}

/// <summary>
/// Rollup of gradient events for a single worker inside a time window.
/// </summary>
public sealed record WorkerTelemetryAggregate(
    string ClientId,
    long TasksCompleted,
    long TokensSeen,
    long WallClockMs,
    double AverageStaleness,
    double AverageLossAfter,
    DateTimeOffset? LastEventUtc);

/// <summary>
/// Rollup of every gradient event inside a time window.
/// </summary>
public sealed record GlobalTelemetryAggregate(
    long TasksCompleted,
    long TokensSeen,
    long WallClockMs,
    double AverageStaleness,
    double AverageLossAfter);
