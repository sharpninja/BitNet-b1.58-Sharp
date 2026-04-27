using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Tests.Logging;

/// <summary>
/// In-memory <see cref="ILogger{T}"/> that captures formatted log messages
/// for test-time assertions. Useful when a test needs to verify that a
/// production code path emitted a specific structured log line (e.g. that
/// per-token decode timing surfaces a non-zero <c>forward_ms</c>).
/// </summary>
public sealed class ListLogger<T> : ILogger<T>
{
    private readonly ConcurrentQueue<LogEntry> _entries = new();

    public IReadOnlyList<LogEntry> Entries => _entries.ToArray();

    public IEnumerable<string> Messages => _entries.Select(e => e.Message);

    public IDisposable BeginScope<TState>(TState state) where TState : notnull => NullScope.Instance;

    public bool IsEnabled(LogLevel logLevel) => logLevel != LogLevel.None;

    public void Log<TState>(
        LogLevel logLevel,
        EventId eventId,
        TState state,
        Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        ArgumentNullException.ThrowIfNull(formatter);
        var message = formatter(state, exception);
        _entries.Enqueue(new LogEntry(logLevel, message, exception));
    }

    public sealed record LogEntry(LogLevel Level, string Message, Exception? Exception);

    private sealed class NullScope : IDisposable
    {
        public static readonly NullScope Instance = new();
        public void Dispose() { }
    }
}
