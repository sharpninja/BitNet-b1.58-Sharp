using System;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.App.Serve.Framing;
using BitNetSharp.App.Serve.Inference;
using BitNetSharp.Core;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

internal static class OllamaChatEndpoints
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapPost("/api/chat", async (HttpContext http, ModelRegistry registry) =>
        {
            var request = await http.Request.ReadJsonAsync<OllamaChatRequest>(http.RequestAborted);
            if (request is null || string.IsNullOrWhiteSpace(request.Model))
            {
                return Results.Json(new OllamaErrorResponse("missing or invalid 'model' field"), ServeJson.Options, statusCode: 400);
            }
            if (request.Messages is null || request.Messages.Count == 0)
            {
                return Results.Json(new OllamaErrorResponse("'messages' must be a non-empty array"), ServeJson.Options, statusCode: 400);
            }
            if (!registry.TryResolve(request.Model, out var entry))
            {
                return Results.Json(new OllamaErrorResponse($"model '{request.Model}' not found, try /api/tags"), ServeJson.Options, statusCode: 404);
            }

            bool stream = request.Stream ?? true;
            var start = DateTimeOffset.UtcNow;
            string prompt = ChatPromptAssembler.Assemble(entry.Model.SystemPrompt, request.Messages);

            int? maxOutputTokens = TryReadMaxOutputTokens(request.Options);

            if (!stream)
            {
                var sb = new System.Text.StringBuilder();
                long ttfbNs = 0;
                bool firstSeen = false;
                await foreach (var piece in entry.Model.StreamResponseAsync(prompt, maxOutputTokens, http.RequestAborted).ConfigureAwait(false))
                {
                    if (!firstSeen && !string.IsNullOrEmpty(piece))
                    {
                        ttfbNs = ServeTimings.ElapsedNanoseconds(start);
                        firstSeen = true;
                    }
                    sb.Append(piece);
                }
                var fullText = sb.ToString();
                return Results.Json(BuildTerminalChat(entry.Card.Name, fullText, start, prompt, fullText, ttfbNs), ServeJson.Options);
            }

            return Results.Stream(async (outputStream) =>
            {
                var writer = new OllamaStreamWriter(outputStream);
                var accumulated = new System.Text.StringBuilder();
                long ttfbNs = 0;
                bool firstSeen = false;
                try
                {
                    // A3: consume the rich per-token stream so each NDJSON
                    // chunk carries forward_ms / select_ms / decode_ms.
                    // Models without native per-token telemetry (default
                    // IHostedAgentModel.StreamTokensAsync impl) surface a
                    // synthetic GeneratedToken with zero timing, in which
                    // case the timing fields stay zero rather than nullable -
                    // we map "exactly zero" back to null on the chunk so
                    // clients can distinguish "no telemetry available" from
                    // "measured 0 ms".
                    await foreach (var token in entry.Model.StreamTokensAsync(prompt, maxOutputTokens, http.RequestAborted).ConfigureAwait(false))
                    {
                        var piece = token.TokenText;
                        if (string.IsNullOrEmpty(piece))
                        {
                            continue;
                        }
                        if (!firstSeen)
                        {
                            ttfbNs = ServeTimings.ElapsedNanoseconds(start);
                            firstSeen = true;
                        }
                        accumulated.Append(piece);
                        var hasTiming = token.ForwardMs > 0d || token.SelectMs > 0d || token.DecodeMs > 0d;
                        var chunk = new OllamaChatResponseChunk(
                            Model: entry.Card.Name,
                            CreatedAt: ServeTimings.UtcNow(),
                            Message: new OllamaChatMessage(Role: "assistant", Content: piece),
                            Done: false,
                            ForwardMs: hasTiming ? token.ForwardMs : null,
                            SelectMs: hasTiming ? token.SelectMs : null,
                            DecodeMs: hasTiming ? token.DecodeMs : null);
                        await writer.WriteAsync(chunk, http.RequestAborted).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (http.RequestAborted.IsCancellationRequested)
                {
                    return;
                }
                await writer.WriteAsync(BuildTerminalChat(entry.Card.Name, string.Empty, start, prompt, accumulated.ToString(), ttfbNs), http.RequestAborted).ConfigureAwait(false);
            }, contentType: "application/x-ndjson");
        });
    }

    private static OllamaChatResponseChunk BuildTerminalChat(string modelName, string visibleContent, DateTimeOffset start, string prompt, string fullResponse, long ttfbNs)
    {
        // Real prefill_ms = time to first emitted token (TTFT). Real eval_ms =
        // remaining wall-clock for the rest of the decode. Both are measured on
        // the server, not estimated.
        long total = ServeTimings.ElapsedNanoseconds(start);
        long prefill = ttfbNs > 0 ? ttfbNs : total;
        long eval = total - prefill;
        if (eval < 0)
        {
            eval = 0;
        }
        return new OllamaChatResponseChunk(
            Model: modelName,
            CreatedAt: ServeTimings.UtcNow(),
            Message: new OllamaChatMessage(Role: "assistant", Content: visibleContent),
            Done: true,
            DoneReason: "stop",
            TotalDuration: total,
            LoadDuration: 0,
            PromptEvalCount: ServeTimings.EstimateTokens(prompt),
            PromptEvalDuration: prefill,
            EvalCount: ServeTimings.EstimateTokens(fullResponse),
            EvalDuration: eval);
    }

    private static int? TryReadMaxOutputTokens(System.Collections.Generic.IReadOnlyDictionary<string, object?>? options)
    {
        if (options is null) return null;
        if (options.TryGetValue("num_predict", out var raw) && raw is not null)
        {
            if (raw is int direct) return direct;
            if (raw is long l) return checked((int)l);
            if (raw is double d) return (int)d;
            if (raw is System.Text.Json.JsonElement elem && elem.TryGetInt32(out int parsed)) return parsed;
        }
        return null;
    }
}
