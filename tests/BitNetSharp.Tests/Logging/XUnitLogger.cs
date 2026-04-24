using System;
using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;

namespace BitNetSharp.Tests.Logging;

/// <summary>
/// Forwards Microsoft.Extensions.Logging output into xUnit's per-test
/// <see cref="ITestOutputHelper"/> so serve/inference logs appear in the
/// test runner's captured output instead of being swallowed.
/// </summary>
public sealed class XUnitLogger : ILogger
{
    private readonly ITestOutputHelper _output;
    private readonly string _category;
    private readonly LogLevel _minLevel;

    public XUnitLogger(ITestOutputHelper output, string category, LogLevel minLevel = LogLevel.Trace)
    {
        _output = output ?? throw new ArgumentNullException(nameof(output));
        _category = category ?? throw new ArgumentNullException(nameof(category));
        _minLevel = minLevel;
    }

    public IDisposable BeginScope<TState>(TState state) where TState : notnull => NullScope.Instance;

    public bool IsEnabled(LogLevel logLevel) => logLevel >= _minLevel && logLevel != LogLevel.None;

    public void Log<TState>(
        LogLevel logLevel,
        EventId eventId,
        TState state,
        Exception? exception,
        Func<TState, Exception?, string> formatter)
    {
        ArgumentNullException.ThrowIfNull(formatter);
        if (!IsEnabled(logLevel))
        {
            return;
        }

        string message = formatter(state, exception);
        string line = $"{DateTime.UtcNow:HH:mm:ss.fff} [{LevelTag(logLevel)}] {_category}: {message}";

        try
        {
            _output.WriteLine(line);
            if (exception is not null)
            {
                _output.WriteLine(exception.ToString());
            }
        }
        catch (InvalidOperationException)
        {
            // ITestOutputHelper throws once the test completes; silently drop
            // late log entries from background work that outlives the test.
        }
    }

    private static string LevelTag(LogLevel level) => level switch
    {
        LogLevel.Trace => "trce",
        LogLevel.Debug => "dbug",
        LogLevel.Information => "info",
        LogLevel.Warning => "warn",
        LogLevel.Error => "fail",
        LogLevel.Critical => "crit",
        _ => "none",
    };

    private sealed class NullScope : IDisposable
    {
        public static readonly NullScope Instance = new();
        public void Dispose() { }
    }
}

public sealed class XUnitLoggerProvider : ILoggerProvider
{
    private readonly ITestOutputHelper _output;
    private readonly LogLevel _minLevel;
    private readonly ConcurrentDictionary<string, XUnitLogger> _loggers = new(StringComparer.Ordinal);

    public XUnitLoggerProvider(ITestOutputHelper output, LogLevel minLevel = LogLevel.Trace)
    {
        _output = output ?? throw new ArgumentNullException(nameof(output));
        _minLevel = minLevel;
    }

    public ILogger CreateLogger(string categoryName) =>
        _loggers.GetOrAdd(categoryName, name => new XUnitLogger(_output, name, _minLevel));

    public void Dispose() => _loggers.Clear();
}

public static class XUnitLoggerExtensions
{
    public static ILoggingBuilder AddXUnit(
        this ILoggingBuilder builder,
        ITestOutputHelper output,
        LogLevel minLevel = LogLevel.Trace)
    {
        ArgumentNullException.ThrowIfNull(builder);
        ArgumentNullException.ThrowIfNull(output);
        builder.AddProvider(new XUnitLoggerProvider(output, minLevel));
        return builder;
    }

    public static ILoggerFactory CreateXUnitLoggerFactory(
        ITestOutputHelper output,
        LogLevel minLevel = LogLevel.Trace) =>
        LoggerFactory.Create(b =>
        {
            b.SetMinimumLevel(minLevel);
            b.AddXUnit(output, minLevel);
        });
}
