namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Uniform error payload returned by every coordinator REST endpoint on
/// the non-2xx paths. Keeping the shape consistent means workers only
/// have to write one deserializer for the sad path.
/// </summary>
/// <param name="Code">Short machine-readable code. Examples:
/// <c>enrollment_key_invalid</c>, <c>unknown_worker</c>,
/// <c>stale_gradient</c>, <c>task_already_assigned</c>.</param>
/// <param name="Message">Human-readable description for worker logs.</param>
public sealed record ErrorResponse(string Code, string Message);
