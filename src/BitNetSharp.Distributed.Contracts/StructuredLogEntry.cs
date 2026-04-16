using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Wire-format DTO for a single structured log entry a worker
/// submits via <c>POST /logs</c>. The coordinator stores these and
/// surfaces them on the admin log viewer page so operators can
/// debug remote workers without SSH access.
/// </summary>
/// <param name="Timestamp">UTC timestamp when the log entry was
/// created on the worker.</param>
/// <param name="Level">Log level: Trace / Debug / Information /
/// Warning / Error / Critical.</param>
/// <param name="Category">Logger category, typically the fully-
/// qualified type name that emitted the log.</param>
/// <param name="Message">Rendered log message.</param>
/// <param name="Exception">Serialized exception string, or null
/// when the log has no associated exception.</param>
/// <param name="WorkerId">Client id of the worker that produced
/// the entry. Filled server-side from the JWT so the worker cannot
/// impersonate another.</param>
public sealed record StructuredLogEntry(
    DateTimeOffset Timestamp,
    string Level,
    string Category,
    string Message,
    string? Exception,
    string? WorkerId);

/// <summary>
/// Batch payload workers POST to <c>/logs</c>. Batching amortizes
/// the per-request overhead so a chatty worker does not hammer the
/// coordinator with individual POSTs.
/// </summary>
public sealed record LogBatch(StructuredLogEntry[] Entries);
