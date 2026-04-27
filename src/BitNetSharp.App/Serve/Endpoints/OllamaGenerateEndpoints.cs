using System;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.App.Serve.Framing;
using BitNetSharp.App.Serve.Inference;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

internal static class OllamaGenerateEndpoints
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapPost("/api/generate", async (HttpContext http, ModelRegistry registry) =>
        {
            var request = await http.Request.ReadJsonAsync<OllamaGenerateRequest>(http.RequestAborted);
            if (request is null || string.IsNullOrWhiteSpace(request.Model))
            {
                return Results.Json(new OllamaErrorResponse("missing or invalid 'model' field"), ServeJson.Options, statusCode: 400);
            }
            if (!registry.TryResolve(request.Model, out var entry))
            {
                return Results.Json(new OllamaErrorResponse($"model '{request.Model}' not found, try /api/tags"), ServeJson.Options, statusCode: 404);
            }

            bool stream = request.Stream ?? true;
            var start = DateTimeOffset.UtcNow;

            string prompt = request.Raw == true
                ? request.Prompt ?? string.Empty
                : PromptTemplate.FlattenHistory(
                    request.System ?? entry.Model.SystemPrompt,
                    new[] { new Microsoft.Extensions.AI.ChatMessage(Microsoft.Extensions.AI.ChatRole.User, request.Prompt ?? string.Empty) });

            int? maxOutputTokens = TryReadMaxOutputTokens(request.Options);

            // Switch to StreamResponseAsync internally so we can capture the
            // real time-to-first-token. Both stream=false and stream=true land
            // here; the only difference is whether intermediate pieces are
            // surfaced over NDJSON or batched into the terminal payload.
            var sb = new System.Text.StringBuilder();
            long ttfbNs = 0;
            bool firstSeen = false;

            if (!stream)
            {
                await foreach (var piece in entry.Model.StreamResponseAsync(prompt, maxOutputTokens, http.RequestAborted).ConfigureAwait(false))
                {
                    if (!firstSeen && !string.IsNullOrEmpty(piece))
                    {
                        ttfbNs = ServeTimings.ElapsedNanoseconds(start);
                        firstSeen = true;
                    }
                    sb.Append(piece);
                }
                return Results.Json(BuildTerminalGenerate(entry.Card.Name, sb.ToString(), start, prompt, ttfbNs), ServeJson.Options);
            }

            return Results.Stream(async (outputStream) =>
            {
                var writer = new OllamaStreamWriter(outputStream);
                try
                {
                    await foreach (var piece in entry.Model.StreamResponseAsync(prompt, maxOutputTokens, http.RequestAborted).ConfigureAwait(false))
                    {
                        if (string.IsNullOrEmpty(piece))
                        {
                            continue;
                        }
                        if (!firstSeen)
                        {
                            ttfbNs = ServeTimings.ElapsedNanoseconds(start);
                            firstSeen = true;
                        }
                        sb.Append(piece);
                        var chunk = new OllamaGenerateResponseChunk(
                            Model: entry.Card.Name,
                            CreatedAt: ServeTimings.UtcNow(),
                            Response: piece,
                            Done: false);
                        await writer.WriteAsync(chunk, http.RequestAborted).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (http.RequestAborted.IsCancellationRequested)
                {
                    return;
                }
                await writer.WriteAsync(BuildTerminalGenerate(entry.Card.Name, string.Empty, start, prompt, ttfbNs), http.RequestAborted).ConfigureAwait(false);
            }, contentType: "application/x-ndjson");
        });
    }

    private static OllamaGenerateResponseChunk BuildTerminalGenerate(string modelName, string response, DateTimeOffset start, string prompt, long ttfbNs)
    {
        // Real prefill_ms = TTFT (time to first emitted piece); real eval_ms =
        // total - prefill. Both come from the actual stream timings.
        long total = ServeTimings.ElapsedNanoseconds(start);
        long prefill = ttfbNs > 0 ? ttfbNs : total;
        long eval = total - prefill;
        if (eval < 0)
        {
            eval = 0;
        }
        return new OllamaGenerateResponseChunk(
            Model: modelName,
            CreatedAt: ServeTimings.UtcNow(),
            Response: response,
            Done: true,
            DoneReason: "stop",
            TotalDuration: total,
            LoadDuration: 0,
            PromptEvalCount: ServeTimings.EstimateTokens(prompt),
            PromptEvalDuration: prefill,
            EvalCount: ServeTimings.EstimateTokens(response),
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
