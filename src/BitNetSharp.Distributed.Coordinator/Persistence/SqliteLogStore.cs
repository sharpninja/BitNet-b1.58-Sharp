using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.Data.Sqlite;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// SQLite-backed store for structured log entries submitted by
/// workers. The admin log viewer page queries this to display
/// recent logs with filtering by level, worker, and search text.
/// </summary>
public sealed class SqliteLogStore : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly object _writeGate = new();

    public SqliteLogStore(string connectionString)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
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
CREATE TABLE IF NOT EXISTS worker_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at INTEGER NOT NULL,
    timestamp   INTEGER NOT NULL,
    level       TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    message     TEXT    NOT NULL,
    exception   TEXT,
    worker_id   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_worker_logs_received
    ON worker_logs(received_at DESC);

CREATE INDEX IF NOT EXISTS ix_worker_logs_worker_level
    ON worker_logs(worker_id, level);
");
    }

    /// <summary>
    /// Inserts a batch of log entries. The <paramref name="workerId"/>
    /// is stamped server-side from the JWT so the worker cannot
    /// impersonate another.
    /// </summary>
    public int InsertBatch(string workerId, IReadOnlyList<LogEntryRow> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);
        if (entries is null || entries.Count == 0)
        {
            return 0;
        }

        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var inserted = 0;

        lock (_writeGate)
        {
            using var transaction = _connection.BeginTransaction();
            using var cmd = _connection.CreateCommand();
            cmd.Transaction = transaction;
            cmd.CommandText = @"
INSERT INTO worker_logs (received_at, timestamp, level, category, message, exception, worker_id)
VALUES ($received_at, $timestamp, $level, $category, $message, $exception, $worker_id);";

            var pReceivedAt = cmd.Parameters.Add("$received_at", SqliteType.Integer);
            var pTimestamp  = cmd.Parameters.Add("$timestamp", SqliteType.Integer);
            var pLevel      = cmd.Parameters.Add("$level", SqliteType.Text);
            var pCategory   = cmd.Parameters.Add("$category", SqliteType.Text);
            var pMessage    = cmd.Parameters.Add("$message", SqliteType.Text);
            var pException  = cmd.Parameters.Add("$exception", SqliteType.Text);
            var pWorkerId   = cmd.Parameters.Add("$worker_id", SqliteType.Text);

            foreach (var entry in entries)
            {
                pReceivedAt.Value = now;
                pTimestamp.Value  = entry.TimestampUnix;
                pLevel.Value     = entry.Level;
                pCategory.Value  = entry.Category;
                pMessage.Value   = entry.Message;
                pException.Value = (object?)entry.Exception ?? DBNull.Value;
                pWorkerId.Value  = workerId;
                cmd.ExecuteNonQuery();
                inserted++;
            }

            transaction.Commit();
        }

        return inserted;
    }

    /// <summary>
    /// Queries recent log entries with optional filtering. Returns
    /// newest first. The <paramref name="limit"/> caps the result
    /// count so the page does not OOM on a chatty fleet.
    /// </summary>
    public IReadOnlyList<LogEntryRow> Query(
        int limit = 200,
        string? workerId = null,
        string? minLevel = null,
        string? search = null)
    {
        var conditions = new List<string>();
        var parameters = new List<(string name, object value)>();

        if (!string.IsNullOrWhiteSpace(workerId))
        {
            conditions.Add("worker_id = $worker_id");
            parameters.Add(("$worker_id", workerId));
        }

        if (!string.IsNullOrWhiteSpace(minLevel))
        {
            conditions.Add("level = $level");
            parameters.Add(("$level", minLevel));
        }

        if (!string.IsNullOrWhiteSpace(search))
        {
            conditions.Add("message LIKE $search");
            parameters.Add(("$search", $"%{search}%"));
        }

        var whereClause = conditions.Count > 0
            ? "WHERE " + string.Join(" AND ", conditions)
            : string.Empty;

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = $@"
SELECT timestamp, level, category, message, exception, worker_id
FROM worker_logs
{whereClause}
ORDER BY id DESC
LIMIT $limit;";
        cmd.Parameters.AddWithValue("$limit", Math.Max(1, Math.Min(limit, 2000)));
        foreach (var (name, value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        var results = new List<LogEntryRow>();
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            results.Add(new LogEntryRow(
                TimestampUnix: reader.GetInt64(0),
                Level: reader.GetString(1),
                Category: reader.GetString(2),
                Message: reader.GetString(3),
                Exception: reader.IsDBNull(4) ? null : reader.GetString(4),
                WorkerId: reader.GetString(5)));
        }

        return results;
    }

    /// <summary>Total row count for diagnostics / tests.</summary>
    public long TotalCount()
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT COUNT(1) FROM worker_logs;";
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
/// Flat row for log entries. Unix timestamp is kept as long to
/// avoid timezone conversions in the store layer; the Blazor page
/// converts to the operator's local time at render.
/// </summary>
public sealed record LogEntryRow(
    long TimestampUnix,
    string Level,
    string Category,
    string Message,
    string? Exception,
    string? WorkerId);
