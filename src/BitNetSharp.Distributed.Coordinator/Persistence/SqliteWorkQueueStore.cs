using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.Data.Sqlite;

namespace BitNetSharp.Distributed.Coordinator.Persistence;

/// <summary>
/// Thin SQLite-backed data access layer for the coordinator work queue.
///
/// The store owns the database connection lifecycle: it opens a single
/// <see cref="SqliteConnection"/> in WAL mode for the process and runs
/// schema migration on first touch. It does NOT enforce business rules
/// — that job belongs to the services that sit on top of it. The store
/// only worries about safe SQL, parameter binding, and row mapping.
///
/// All state transitions are wrapped in <c>BEGIN IMMEDIATE</c>
/// transactions so concurrent worker requests cannot race each other
/// into an inconsistent state. SQLite's single-writer guarantee plus
/// WAL concurrent readers is exactly what the single-coordinator
/// topology needs.
/// </summary>
public sealed class SqliteWorkQueueStore : IDisposable
{
    private readonly SqliteConnection _connection;
    private readonly TimeProvider _time;
    private readonly object _writeGate = new();

    public SqliteWorkQueueStore(string connectionString, TimeProvider? time = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        _time = time ?? TimeProvider.System;
        _connection = new SqliteConnection(connectionString);
        _connection.Open();

        // WAL + normal sync is the standard recipe for "safe but fast"
        // single-writer workloads. busy_timeout gives short retries a
        // chance to land instead of bombing out on SQLITE_BUSY.
        ExecuteNonQuery("PRAGMA journal_mode = WAL;");
        ExecuteNonQuery("PRAGMA synchronous = NORMAL;");
        ExecuteNonQuery("PRAGMA busy_timeout = 5000;");
        ExecuteNonQuery("PRAGMA foreign_keys = ON;");

        MigrateSchema();
    }

    /// <summary>
    /// Creates the schema if it does not exist. Idempotent — safe to run
    /// on every process start. Migrations are tracked by a single
    /// <c>schema_version</c> row so future upgrades can branch on it.
    /// </summary>
    private void MigrateSchema()
    {
        ExecuteNonQuery(@"
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1');

CREATE TABLE IF NOT EXISTS tasks (
    task_id          TEXT    PRIMARY KEY,
    weight_version   INTEGER NOT NULL,
    shard_id         TEXT    NOT NULL,
    shard_offset     INTEGER NOT NULL,
    shard_length     INTEGER NOT NULL,
    tokens_per_task  INTEGER NOT NULL,
    k_local_steps    INTEGER NOT NULL,
    hp_json          TEXT    NOT NULL,
    state            TEXT    NOT NULL,
    assigned_to      TEXT,
    assigned_at      INTEGER,
    deadline_at      INTEGER,
    attempt          INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL,
    completed_at     INTEGER
);

CREATE INDEX IF NOT EXISTS ix_tasks_state_created
    ON tasks(state, created_at);

CREATE INDEX IF NOT EXISTS ix_tasks_state_deadline
    ON tasks(state, deadline_at);

CREATE INDEX IF NOT EXISTS ix_tasks_assigned_to
    ON tasks(assigned_to);
");
    }

    /// <summary>
    /// Inserts a new task in the <c>Pending</c> state. The task is
    /// immediately visible to the next dequeue call.
    /// </summary>
    public void EnqueuePending(WorkTaskRecord task)
    {
        ArgumentNullException.ThrowIfNull(task);
        if (task.State != WorkTaskState.Pending)
        {
            throw new ArgumentException("Enqueued tasks must start in the Pending state.", nameof(task));
        }

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
INSERT INTO tasks (
    task_id, weight_version, shard_id, shard_offset, shard_length,
    tokens_per_task, k_local_steps, hp_json, state,
    assigned_to, assigned_at, deadline_at, attempt, created_at, completed_at
) VALUES (
    $task_id, $weight_version, $shard_id, $shard_offset, $shard_length,
    $tokens_per_task, $k_local_steps, $hp_json, 'Pending',
    NULL, NULL, NULL, 0, $created_at, NULL
);";
            cmd.Parameters.AddWithValue("$task_id", task.TaskId);
            cmd.Parameters.AddWithValue("$weight_version", task.WeightVersion);
            cmd.Parameters.AddWithValue("$shard_id", task.ShardId);
            cmd.Parameters.AddWithValue("$shard_offset", task.ShardOffset);
            cmd.Parameters.AddWithValue("$shard_length", task.ShardLength);
            cmd.Parameters.AddWithValue("$tokens_per_task", task.TokensPerTask);
            cmd.Parameters.AddWithValue("$k_local_steps", task.KLocalSteps);
            cmd.Parameters.AddWithValue("$hp_json", task.HyperparametersJson);
            cmd.Parameters.AddWithValue("$created_at", task.CreatedAtUtc.ToUnixTimeSeconds());
            cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Atomically claims the oldest pending task for the given worker,
    /// sets its state to <c>Assigned</c>, stamps the deadline, and
    /// returns the updated row. Returns <c>null</c> when the queue is
    /// empty. The whole select-and-update runs inside
    /// <c>BEGIN IMMEDIATE</c> so concurrent workers cannot claim the
    /// same task.
    /// </summary>
    public WorkTaskRecord? TryClaimNextPending(string workerId, TimeSpan leaseDuration)
        => TryClaimNextPending(workerId, _ => leaseDuration);

    /// <summary>
    /// Claim variant that lets the caller size the lease per-task by
    /// inspecting <c>tokens_per_task</c>. Used by
    /// <c>ClaimNextTaskCommandHandler</c> to set a deadline based on
    /// the worker's measured real-training throughput instead of a
    /// fixed multiple of the calibration-time target duration.
    /// </summary>
    public WorkTaskRecord? TryClaimNextPending(string workerId, Func<long, TimeSpan> leaseForTokensPerTask)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);
        ArgumentNullException.ThrowIfNull(leaseForTokensPerTask);

        lock (_writeGate)
        {
            // Microsoft.Data.Sqlite's default BeginTransaction() maps to
            // BEGIN IMMEDIATE (non-deferred, Serializable), which is what
            // we need to prevent two dequeue callers from selecting the
            // same pending row.
            using var transaction = _connection.BeginTransaction();

            using var selectCmd = _connection.CreateCommand();
            selectCmd.Transaction = transaction;
            selectCmd.CommandText = @"
SELECT task_id, tokens_per_task FROM tasks
WHERE state = 'Pending'
ORDER BY created_at ASC, task_id ASC
LIMIT 1;";
            using var selReader = selectCmd.ExecuteReader();
            if (!selReader.Read())
            {
                selReader.Close();
                transaction.Commit();
                return null;
            }
            var taskId = selReader.GetString(0);
            var tokensPerTask = selReader.GetInt64(1);
            selReader.Close();

            var leaseDuration = leaseForTokensPerTask(tokensPerTask);
            if (leaseDuration <= TimeSpan.Zero)
            {
                throw new InvalidOperationException("Lease calculator returned a non-positive duration.");
            }

            var nowUnix = _time.GetUtcNow().ToUnixTimeSeconds();
            var deadlineUnix = _time.GetUtcNow().Add(leaseDuration).ToUnixTimeSeconds();

            using var updateCmd = _connection.CreateCommand();
            updateCmd.Transaction = transaction;
            updateCmd.CommandText = @"
UPDATE tasks
SET state       = 'Assigned',
    assigned_to = $worker,
    assigned_at = $now,
    deadline_at = $deadline,
    attempt     = attempt + 1
WHERE task_id = $task_id AND state = 'Pending';";
            updateCmd.Parameters.AddWithValue("$worker", workerId);
            updateCmd.Parameters.AddWithValue("$now", nowUnix);
            updateCmd.Parameters.AddWithValue("$deadline", deadlineUnix);
            updateCmd.Parameters.AddWithValue("$task_id", taskId);
            var rows = updateCmd.ExecuteNonQuery();
            if (rows != 1)
            {
                // Lost the race with another thread that re-released
                // the row between SELECT and UPDATE. Treat as empty.
                transaction.Commit();
                return null;
            }

            var record = LoadById(taskId, transaction);
            transaction.Commit();
            return record;
        }
    }

    /// <summary>
    /// Marks the given task <c>Done</c>. Returns <c>true</c> if the row
    /// transitioned, <c>false</c> when the task does not exist or was
    /// not currently assigned to the caller.
    /// </summary>
    public bool MarkCompleted(string taskId, string workerId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(taskId);
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
UPDATE tasks
SET state        = 'Done',
    completed_at = $now
WHERE task_id = $task_id
  AND assigned_to = $worker
  AND state = 'Assigned';";
            cmd.Parameters.AddWithValue("$now", _time.GetUtcNow().ToUnixTimeSeconds());
            cmd.Parameters.AddWithValue("$task_id", taskId);
            cmd.Parameters.AddWithValue("$worker", workerId);
            return cmd.ExecuteNonQuery() == 1;
        }
    }

    /// <summary>
    /// Marks the given task <c>Failed</c> permanently. Only allowed if
    /// the task is currently <c>Assigned</c> to <paramref name="workerId"/>.
    /// </summary>
    public bool MarkFailed(string taskId, string workerId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(taskId);
        ArgumentException.ThrowIfNullOrWhiteSpace(workerId);

        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
UPDATE tasks
SET state        = 'Failed',
    completed_at = $now
WHERE task_id = $task_id
  AND assigned_to = $worker
  AND state = 'Assigned';";
            cmd.Parameters.AddWithValue("$now", _time.GetUtcNow().ToUnixTimeSeconds());
            cmd.Parameters.AddWithValue("$task_id", taskId);
            cmd.Parameters.AddWithValue("$worker", workerId);
            return cmd.ExecuteNonQuery() == 1;
        }
    }

    /// <summary>
    /// Finds every <c>Assigned</c> task whose deadline has passed and
    /// returns it to <c>Pending</c> so another worker can pick it up.
    /// Returns the number of tasks that were recycled.
    /// </summary>
    public int RecycleTimedOutAssignments()
    {
        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = @"
UPDATE tasks
SET state       = 'Pending',
    assigned_to = NULL,
    assigned_at = NULL,
    deadline_at = NULL
WHERE state = 'Assigned'
  AND deadline_at IS NOT NULL
  AND deadline_at < $now;";
            cmd.Parameters.AddWithValue("$now", _time.GetUtcNow().ToUnixTimeSeconds());
            return cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Deletes every row in the given states. Returns rows affected.
    /// Admin/CLI use — wipes seeded but never-picked-up tasks when
    /// the seed parameters themselves were wrong (e.g. KLocalSteps
    /// mis-set) without touching completed history.
    /// </summary>
    public int DeleteByStates(IReadOnlyList<WorkTaskState> states)
    {
        ArgumentNullException.ThrowIfNull(states);
        if (states.Count == 0) return 0;
        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            var placeholders = new List<string>(states.Count);
            for (var i = 0; i < states.Count; i++)
            {
                var p = $"$s{i}";
                placeholders.Add(p);
                cmd.Parameters.AddWithValue(p, states[i].ToString());
            }
            cmd.CommandText = $"DELETE FROM tasks WHERE state IN ({string.Join(",", placeholders)});";
            return cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Counts tasks whose <c>task_id</c> begins with the given literal
    /// prefix and which are in <paramref name="state"/>. Companion to
    /// <see cref="DeleteByTaskIdPrefixAndState"/>: lets a destructive
    /// CLI show a dry-run count before the user authorises the delete.
    /// </summary>
    public int CountByTaskIdPrefixAndState(string taskIdPrefix, WorkTaskState state)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(taskIdPrefix);
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT COUNT(1) FROM tasks WHERE state = $state AND task_id LIKE $prefix;";
        cmd.Parameters.AddWithValue("$state", state.ToString());
        cmd.Parameters.AddWithValue("$prefix", taskIdPrefix + "%");
        var result = cmd.ExecuteScalar();
        return result is null or DBNull ? 0 : Convert.ToInt32(result, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Deletes tasks whose <c>task_id</c> begins with the given literal
    /// prefix AND are in the given lifecycle state. Used by the
    /// <c>purge-legacy-seed-rows</c> CLI to remove the historical
    /// <c>task-seed-*</c> synthetic Done rows so the dashboard progress
    /// bar tracks real-corpus training signal only. Returns rows
    /// affected.
    /// </summary>
    public int DeleteByTaskIdPrefixAndState(string taskIdPrefix, WorkTaskState state)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(taskIdPrefix);
        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM tasks WHERE state = $state AND task_id LIKE $prefix;";
            cmd.Parameters.AddWithValue("$state", state.ToString());
            cmd.Parameters.AddWithValue("$prefix", taskIdPrefix + "%");
            return cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Deletes pending tasks whose shard_id begins with the given
    /// literal prefix. Used by the <c>purge-shards</c> CLI to remove
    /// an abandoned corpus (e.g. v1) from the queue while leaving
    /// in-flight Assigned rows alone.
    /// </summary>
    public int DeletePendingByShardPrefix(string shardPrefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shardPrefix);
        lock (_writeGate)
        {
            using var cmd = _connection.CreateCommand();
            cmd.CommandText = "DELETE FROM tasks WHERE state = 'Pending' AND shard_id LIKE $prefix;";
            cmd.Parameters.AddWithValue("$prefix", shardPrefix + "%");
            return cmd.ExecuteNonQuery();
        }
    }

    /// <summary>
    /// Returns one entry per worker that currently owns an Assigned
    /// task. Keyed by <c>assigned_to</c>; value carries the task id and
    /// the UTC instant the claim landed. Used by the dashboard's
    /// per-worker live view so each row can show which task is in
    /// flight and how long it has been running.
    /// </summary>
    public IReadOnlyDictionary<string, AssignedTaskInfo> ListAssignedByWorker()
    {
        var results = new Dictionary<string, AssignedTaskInfo>(StringComparer.Ordinal);
        using var cmd = _connection.CreateCommand();
        // A worker may technically have more than one Assigned row if
        // a lease got extended manually; show the most recently
        // assigned one.
        cmd.CommandText = @"
SELECT assigned_to, task_id, assigned_at
FROM tasks
WHERE state = 'Assigned' AND assigned_to IS NOT NULL
ORDER BY assigned_at DESC;";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var clientId = reader.GetString(0);
            if (results.ContainsKey(clientId))
            {
                continue;
            }
            var taskId = reader.GetString(1);
            var assignedAt = reader.IsDBNull(2)
                ? (DateTimeOffset?)null
                : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(2));
            results[clientId] = new AssignedTaskInfo(taskId, assignedAt);
        }
        return results;
    }

    /// <summary>
    /// Lists tasks in the given states, most recent first. Admin-facing
    /// helper that powers the task browser page — callers pass
    /// <c>new[] { Pending, Assigned }</c> for the queue view or
    /// <c>new[] { Done, Failed }</c> for the history view. Ordering
    /// keys off <c>completed_at DESC</c> then <c>assigned_at DESC</c>
    /// then <c>created_at DESC</c> so finished tasks float the newest
    /// result, active claims float the newest assignment, and pending
    /// entries float the newest enqueue.
    /// </summary>
    public IReadOnlyList<WorkTaskRecord> ListByStates(IReadOnlyList<WorkTaskState> states, int limit = 200)
    {
        ArgumentNullException.ThrowIfNull(states);
        if (states.Count == 0)
        {
            return Array.Empty<WorkTaskRecord>();
        }
        if (limit <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(limit), limit, "limit must be positive");
        }

        using var cmd = _connection.CreateCommand();
        // Build parameterised IN list so SQL injection is impossible.
        var placeholders = new string[states.Count];
        for (var i = 0; i < states.Count; i++)
        {
            var p = $"$s{i}";
            placeholders[i] = p;
            cmd.Parameters.AddWithValue(p, states[i].ToString());
        }
        cmd.CommandText = @"
SELECT task_id, weight_version, shard_id, shard_offset, shard_length,
       tokens_per_task, k_local_steps, hp_json, state,
       assigned_to, assigned_at, deadline_at, attempt,
       created_at, completed_at
FROM tasks
WHERE state IN (" + string.Join(", ", placeholders) + @")
ORDER BY COALESCE(completed_at, 0) DESC,
         COALESCE(assigned_at,  0) DESC,
         created_at DESC,
         task_id DESC
LIMIT $limit;";
        cmd.Parameters.AddWithValue("$limit", limit);

        var results = new List<WorkTaskRecord>(limit);
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            results.Add(new WorkTaskRecord(
                TaskId: reader.GetString(0),
                WeightVersion: reader.GetInt64(1),
                ShardId: reader.GetString(2),
                ShardOffset: reader.GetInt64(3),
                ShardLength: reader.GetInt64(4),
                TokensPerTask: reader.GetInt64(5),
                KLocalSteps: reader.GetInt32(6),
                HyperparametersJson: reader.GetString(7),
                State: Enum.Parse<WorkTaskState>(reader.GetString(8)),
                AssignedWorkerId: reader.IsDBNull(9) ? null : reader.GetString(9),
                AssignedAtUtc: reader.IsDBNull(10) ? null : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(10)),
                DeadlineUtc: reader.IsDBNull(11) ? null : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(11)),
                Attempt: reader.GetInt32(12),
                CreatedAtUtc: DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(13)),
                CompletedAtUtc: reader.IsDBNull(14) ? null : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(14))));
        }
        return results;
    }

    /// <summary>
    /// Counts Assigned tasks whose lease deadline has passed but whose
    /// owning worker is still sending heartbeats inside
    /// <paramref name="staleAfter"/>. This is the "soft-expired but
    /// alive" signal: the task is technically eligible for recycling by
    /// <see cref="RecycleTimedOutAssignments"/>, yet the worker is
    /// demonstrably still on the network — meaning real backprop is
    /// likely running longer than the lease.
    ///
    /// <para>
    /// The dashboard pairs this counter with Assigned so the operator
    /// can tell the difference between a stuck worker (heartbeat stale)
    /// and a slow-but-alive worker (heartbeat fresh, deadline passed).
    /// JOINs the <c>workers</c> table on <c>assigned_to = worker_id</c>
    /// because both stores share the same SQLite file.
    /// </para>
    /// </summary>
    public int CountSoftExpiredButAlive(TimeSpan staleAfter)
    {
        if (staleAfter <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(staleAfter), "Stale threshold must be positive.");
        }
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT COUNT(1)
FROM tasks t
INNER JOIN workers w ON t.assigned_to = w.worker_id
WHERE t.state = 'Assigned'
  AND t.deadline_at IS NOT NULL
  AND t.deadline_at < $now
  AND w.last_heartbeat >= $cutoff;";
        var nowUnix = _time.GetUtcNow().ToUnixTimeSeconds();
        var cutoff = _time.GetUtcNow().Subtract(staleAfter).ToUnixTimeSeconds();
        cmd.Parameters.AddWithValue("$now", nowUnix);
        cmd.Parameters.AddWithValue("$cutoff", cutoff);
        var result = cmd.ExecuteScalar();
        return result is null or DBNull ? 0 : Convert.ToInt32(result, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Counts Assigned tasks whose lease deadline has passed AND whose
    /// owning worker's heartbeat is also stale (older than
    /// <paramref name="staleAfter"/>). The companion counter to
    /// <see cref="CountSoftExpiredButAlive"/>: both share the deadline
    /// predicate, but this one fires when the worker has gone silent —
    /// a genuine stuck/dead signal the operator should act on
    /// (restart container, check node health).
    ///
    /// <para>
    /// Also counts Assigned tasks whose <c>assigned_to</c> worker id
    /// has no row in <c>workers</c> at all, since that means the
    /// coordinator lost track of the worker entirely.
    /// </para>
    /// </summary>
    public int CountStuckDead(TimeSpan staleAfter)
    {
        if (staleAfter <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(staleAfter), "Stale threshold must be positive.");
        }
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = @"
SELECT COUNT(1)
FROM tasks t
LEFT JOIN workers w ON t.assigned_to = w.worker_id
WHERE t.state = 'Assigned'
  AND t.deadline_at IS NOT NULL
  AND t.deadline_at < $now
  AND (w.worker_id IS NULL OR w.last_heartbeat < $cutoff);";
        var nowUnix = _time.GetUtcNow().ToUnixTimeSeconds();
        var cutoff = _time.GetUtcNow().Subtract(staleAfter).ToUnixTimeSeconds();
        cmd.Parameters.AddWithValue("$now", nowUnix);
        cmd.Parameters.AddWithValue("$cutoff", cutoff);
        var result = cmd.ExecuteScalar();
        return result is null or DBNull ? 0 : Convert.ToInt32(result, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Returns the count of tasks currently in the given state. Handy
    /// for <c>/status</c> dashboards and smoke tests.
    /// </summary>
    public int CountByState(WorkTaskState state)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT COUNT(1) FROM tasks WHERE state = $state;";
        cmd.Parameters.AddWithValue("$state", state.ToString());
        var result = cmd.ExecuteScalar();
        return result is null or DBNull ? 0 : Convert.ToInt32(result, CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// Reads a single task row by id. Returns <c>null</c> if no row
    /// exists. Public so tests and the coordinator's <c>/status</c>
    /// endpoint can inspect task state directly.
    /// </summary>
    public WorkTaskRecord? GetById(string taskId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(taskId);
        return LoadById(taskId, transaction: null);
    }

    private WorkTaskRecord? LoadById(string taskId, SqliteTransaction? transaction)
    {
        using var cmd = _connection.CreateCommand();
        cmd.Transaction = transaction;
        cmd.CommandText = @"
SELECT task_id, weight_version, shard_id, shard_offset, shard_length,
       tokens_per_task, k_local_steps, hp_json, state,
       assigned_to, assigned_at, deadline_at, attempt,
       created_at, completed_at
FROM tasks WHERE task_id = $task_id;";
        cmd.Parameters.AddWithValue("$task_id", taskId);
        using var reader = cmd.ExecuteReader();
        if (!reader.Read())
        {
            return null;
        }

        return new WorkTaskRecord(
            TaskId: reader.GetString(0),
            WeightVersion: reader.GetInt64(1),
            ShardId: reader.GetString(2),
            ShardOffset: reader.GetInt64(3),
            ShardLength: reader.GetInt64(4),
            TokensPerTask: reader.GetInt64(5),
            KLocalSteps: reader.GetInt32(6),
            HyperparametersJson: reader.GetString(7),
            State: Enum.Parse<WorkTaskState>(reader.GetString(8)),
            AssignedWorkerId: reader.IsDBNull(9) ? null : reader.GetString(9),
            AssignedAtUtc: reader.IsDBNull(10) ? null : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(10)),
            DeadlineUtc: reader.IsDBNull(11) ? null : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(11)),
            Attempt: reader.GetInt32(12),
            CreatedAtUtc: DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(13)),
            CompletedAtUtc: reader.IsDBNull(14) ? null : DateTimeOffset.FromUnixTimeSeconds(reader.GetInt64(14)));
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
/// A snapshot of the task a worker currently holds an Assigned lease
/// on. Used by the dashboard to render per-worker live progress.
/// </summary>
public sealed record AssignedTaskInfo(string TaskId, DateTimeOffset? AssignedAtUtc);
