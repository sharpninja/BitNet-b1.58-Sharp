using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.Data.Sqlite;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// SQLite-backed data access layer for the coordinator's <c>workers</c>
/// table. Each row is keyed by the Duende IdentityServer
/// <c>client_id</c> the worker authenticates as, so registering a
/// worker is effectively "attach this capability report to the
/// already-authenticated OAuth client".
///
/// Opens and owns its own WAL-mode connection against the coordinator
/// database file so it can coexist with
/// <see cref="SqliteWorkQueueStore"/>; SQLite's internal locking
/// serializes concurrent writers safely.
/// </summary>
public sealed class SqliteWorkerRegistryStore : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly TimeProvider _time;
    private readonly object _writeGate = new();

    public SqliteWorkerRegistryStore(string connectionString, TimeProvider? time = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        _time = time ?? TimeProvider.System;
        _connection = new SqliteConnection(connectionString);
        _connection.Open();

        ExecuteNonQuery("PRAGMA journal_mode = WAL;");
        ExecuteNonQuery("PRAGMA synchronous = NORMAL;");
        ExecuteNonQuery("PRAGMA busy_timeout = 5000;");
        ExecuteNonQuery("PRAGMA foreign_keys = ON;");

        MigrateSchema();
    }

    private void MigrateSchema()
    {
        ExecuteNonQuery(@"
CREATE TABLE IF NOT EXISTS workers (
    worker_id                   TEXT    PRIMARY KEY,
    name                        TEXT    NOT NULL,
    cpu_threads                 INTEGER NOT NULL,
    tokens_per_sec              REAL    NOT NULL,
    recommended_tokens_per_task INTEGER NOT NULL,
    process_architecture        TEXT,
    os_description              TEXT,
    registered_at               INTEGER NOT NULL,
    last_heartbeat              INTEGER NOT NULL,
    state                       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_workers_heartbeat
    ON workers(last_heartbeat);

CREATE INDEX IF NOT EXISTS ix_workers_state
    ON workers(state);
");
    }

    /// <summary>
    /// Inserts a newly registered worker, or updates the existing row
    /// in place if the coordinator is seeing a re-registration from the
    /// same authenticated Duende client (which is the normal case after
    /// a worker container restarts). The row's lifecycle state is reset
    /// to <see cref="WorkerState.Active"/> on upsert so a previously
    /// "Gone" worker can rejoin cleanly.
    /// </summary>
    public void Upsert(WorkerRecord worker)
    {
        ArgumentNullException.ThrowIfNull(worker);

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
INSERT INTO workers (
    worker_id, name, cpu_threads, tokens_per_sec,
    recommended_tokens_per_task, process_architecture, os_description,
    registered_at, last_heartbeat, state
) VALUES (
    $worker_id, $name, $cpu_threads, $tokens_per_sec,
    $recommended_tokens_per_task, $process_architecture, $os_description,
    $registered_at, $last_heartbeat, $state
)
ON CONFLICT(worker_id) DO UPDATE SET
    name                        = excluded.name,
    cpu_threads                 = excluded.cpu_threads,
    tokens_per_sec              = excluded.tokens_per_sec,
    recommended_tokens_per_task = excluded.recommended_tokens_per_task,
    process_architecture        = excluded.process_architecture,
    os_description              = excluded.os_description,
    registered_at               = excluded.registered_at,
    last_heartbeat              = excluded.last_heartbeat,
    state                       = excluded.state;";
            cmd.Parameters.AddWithValue("$worker_id", worker.WorkerId);
            cmd.Parameters.AddWithValue("$name", worker.Name);
            cmd.Parameters.AddWithValue("$cpu_threads", worker.CpuThreads);
            cmd.Parameters.AddWithValue("$tokens_per_sec", worker.TokensPerSecond);
            cmd.Parameters.AddWithValue("$recommended_tokens_per_task", worker.RecommendedTokensPerTask);
            cmd.Parameters.AddWithValue("$process_architecture", (object?)worker.ProcessArchitecture ?? DBNull.Value);
            cmd.Parameters.AddWithValue("$os_description", (object?)worker.OsDescription ?? DBNull.Value);
            cmd.Parameters.AddWithValue("$registered_at", worker.RegisteredAtUtc.ToUnixTimeSeconds());
            cmd.Parameters.AddWithValue("$last_heartbeat", worker.LastHeartbeatUtc.ToUnixTimeSeconds());
            cmd.Parameters.AddWithValue("$state", worker.State.ToString());
            cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Finds a worker by its opaque id (equal to the Duende client_id).
    /// Returns <c>null</c> if no such worker exists.
    /// </summary>
    public WorkerRecord? FindById(string workerId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);
        return Load("worker_id = $id", ("$id", workerId));
    }

    /// <summary>
    /// Updates the worker's last-heartbeat timestamp to "now" from the
    /// injected <see cref="TimeProvider"/>. Returns <c>true</c> if a row
    /// was updated, <c>false</c> if the worker id does not exist.
    /// </summary>
    public bool TouchHeartbeat(string workerId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
UPDATE workers
SET last_heartbeat = $now
WHERE worker_id = $id;";
            cmd.Parameters.AddWithValue("$now", _time.GetUtcNow().ToUnixTimeSeconds());
            cmd.Parameters.AddWithValue("$id", workerId);
            return cmd.ExecuteNonQuery() == 1;
        }
    }

    /// <summary>
    /// Transitions a worker to <see cref="WorkerState.Draining"/> so
    /// the coordinator stops assigning it new work. Safe to re-run.
    /// </summary>
    public bool MarkDraining(string workerId) => UpdateState(workerId, WorkerState.Draining);

    /// <summary>
    /// Transitions a worker to <see cref="WorkerState.Gone"/>. Invoked
    /// when heartbeats go silent past the deadline or when a worker
    /// explicitly deregisters.
    /// </summary>
    public bool MarkGone(string workerId) => UpdateState(workerId, WorkerState.Gone);

    /// <summary>
    /// Counts workers currently in the given lifecycle state. Used by
    /// the <c>/status</c> dashboard.
    /// </summary>
    public int CountByState(WorkerState state)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT COUNT(1) FROM workers WHERE state = $state;";
        cmd.Parameters.AddWithValue("$state", state.ToString());
        var result = cmd.ExecuteScalar();
        return result is null or DBNull ? 0 : Convert.ToInt32(result, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Finds every <see cref="WorkerState.Active"/> worker whose last
    /// heartbeat is older than <paramref name="staleAfter"/> relative to
    /// the injected clock and transitions them to
    /// <see cref="WorkerState.Gone"/>. Returns the count of transitions.
    /// </summary>
    public int SweepStaleWorkers(TimeSpan staleAfter)
    {
        if (staleAfter <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(staleAfter), "Stale threshold must be positive.");
        }

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
UPDATE workers
SET state = 'Gone'
WHERE state = 'Active'
  AND last_heartbeat < $cutoff;";
            var cutoff = _time.GetUtcNow().Subtract(staleAfter).ToUnixTimeSeconds();
            cmd.Parameters.AddWithValue("$cutoff", cutoff);
            return cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Enumerates every row in the <c>workers</c> table in registration
    /// order. Kept simple for the v1 status dashboard; larger fleets
    /// will add pagination later.
    /// </summary>
    public IReadOnlyList<WorkerRecord> ListAll()
    {
        var results = new List<WorkerRecord>();
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT worker_id, name, cpu_threads, tokens_per_sec,
       recommended_tokens_per_task, process_architecture, os_description,
       registered_at, last_heartbeat, state
FROM workers
ORDER BY registered_at ASC, worker_id ASC;";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            results.Add(MapRow(reader));
        }

        return results;
    }

    private bool UpdateState(string workerId, WorkerState state)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "UPDATE workers SET state = $state WHERE worker_id = $id;";
            cmd.Parameters.AddWithValue("$state", state.ToString());
            cmd.Parameters.AddWithValue("$id", workerId);
            return cmd.ExecuteNonQuery() == 1;
        }
    }

    private WorkerRecord? Load(string whereClause, params (string name, object value)[] parameters)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = $@"
SELECT worker_id, name, cpu_threads, tokens_per_sec,
       recommended_tokens_per_task, process_architecture, os_description,
       registered_at, last_heartbeat, state
FROM workers WHERE {whereClause};";
        foreach (var (name, value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        using var reader = cmd.ExecuteReader();
        return reader.Read() ? MapRow(reader) : null;
    }

    private static WorkerRecord MapRow(SqliteDataReader reader)
    {
        return new WorkerRecord(
            WorkerId: reader.GetString(0),
            Name: reader.GetString(1),
            CpuThreads: reader.GetInt32(2),
            TokensPerSecond: reader.GetDouble(3),
            RecommendedTokensPerTask: reader.GetInt64(4),
            ProcessArchitecture: reader.IsDBNull(5) ? null : reader.GetString(5),
            OsDescription: reader.IsDBNull(6) ? null : reader.GetString(6),
            RegisteredAtUtc: DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(7)),
            LastHeartbeatUtc: DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(8)),
            State: Enum.Parse<WorkerState>(reader.GetString(9)));
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
