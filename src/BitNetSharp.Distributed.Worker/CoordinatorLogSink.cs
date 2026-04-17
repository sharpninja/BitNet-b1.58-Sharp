using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using Serilog.Events;
using Serilog.Formatting.Display;
using Serilog.Sinks.PeriodicBatching;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Serilog periodic-batching sink that ships structured log events
/// to the coordinator's <c>POST /logs</c> endpoint as a
/// <see cref="LogBatch"/>. Piggybacks on the shared
/// <see cref="CoordinatorClient"/> so the X-Api-Key + X-Worker-Id
/// headers the sink relies on are applied by the client's
/// default-request-header configuration — the sink itself never
/// touches credentials.
///
/// <para>
/// The sink is wired up early — before the worker has called
/// <c>/register</c> — so it buffers silently until
/// <see cref="SetClient"/> is called with a live
/// <see cref="CoordinatorClient"/>. Any log events emitted before
/// that point are dropped (they still go to the parallel console
/// sink).
/// </para>
///
/// <para>
/// The sink never throws into the Serilog pipeline on transient HTTP
/// failures; it just swallows them so a network blip does not crash
/// the worker. The next batch retries automatically because
/// <see cref="PeriodicBatchingSink"/> re-invokes
/// <see cref="EmitBatchAsync"/> on the next timer tick.
/// </para>
/// </summary>
internal sealed class CoordinatorLogSink : IBatchedLogEventSink, IDisposable
{
    private readonly MessageTemplateTextFormatter _formatter;
    private volatile CoordinatorClient? _client;

    /// <summary>
    /// Constructs the sink with a message template formatter.
    /// <paramref name="outputTemplate"/> follows the standard
    /// Serilog template syntax; the default mirrors the console
    /// sink's output so logs look the same in both destinations.
    /// </summary>
    public CoordinatorLogSink(string outputTemplate = "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
    {
        _formatter = new MessageTemplateTextFormatter(outputTemplate);
    }

    /// <summary>
    /// Called by the worker's startup path once registration succeeds.
    /// All batches emitted before this call are silently dropped.
    /// </summary>
    public void SetClient(CoordinatorClient client)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
    }

    public async Task EmitBatchAsync(IEnumerable<LogEvent> batch)
    {
        var client = _client;
        var entries = batch.Select(MapEvent).ToArray();
        if (client is null || entries.Length == 0)
        {
            return;
        }
        try
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, "logs")
            {
                Content = JsonContent.Create(new LogBatch(entries))
            };
            // No explicit auth header needed — the CoordinatorClient's
            // default request headers (X-Api-Key + X-Worker-Id) are
            // applied to every outbound request, including this one.
            await client.SendRawAsync(request).ConfigureAwait(false);
        }
        catch
        {
            // Swallow — the console sink has the entries regardless.
        }
    }

    public Task OnEmptyBatchAsync() => Task.CompletedTask;

    public void Dispose()
    {
        // No resources to release — CoordinatorClient is owned by
        // the worker's main lifetime scope.
    }

    private StructuredLogEntry MapEvent(LogEvent logEvent)
    {
        using var writer = new StringWriter();
        _formatter.Format(logEvent, writer);
        var rendered = writer.ToString().TrimEnd();

        string? exception = null;
        if (logEvent.Exception is not null)
        {
            exception = logEvent.Exception.ToString();
        }

        return new StructuredLogEntry(
            Timestamp: logEvent.Timestamp,
            Level: logEvent.Level.ToString(),
            Category: logEvent.Properties.TryGetValue("SourceContext", out var ctx)
                ? ctx.ToString().Trim('"')
                : string.Empty,
            Message: rendered,
            Exception: exception,
            WorkerId: null); // Stamped server-side by the coordinator.
    }
}
