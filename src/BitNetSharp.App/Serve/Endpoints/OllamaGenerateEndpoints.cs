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
            var response = await entry.Model.GetResponseAsync(prompt, maxOutputTokens, http.RequestAborted).ConfigureAwait(false);

            if (!stream)
            {
                return Results.Json(BuildTerminalGenerate(entry.Card.Name, response.Text, start, prompt), ServeJson.Options);
            }

            return Results.Stream(async (stream) =>
            {
                var writer = new OllamaStreamWriter(stream);
                if (!string.IsNullOrEmpty(response.Text))
                {
                    var chunk = new OllamaGenerateResponseChunk(
                        Model: entry.Card.Name,
                        CreatedAt: ServeTimings.UtcNow(),
                        Response: response.Text,
                        Done: false);
                    await writer.WriteAsync(chunk, http.RequestAborted).ConfigureAwait(false);
                }
                await writer.WriteAsync(BuildTerminalGenerate(entry.Card.Name, string.Empty, start, prompt), http.RequestAborted).ConfigureAwait(false);
            }, contentType: "application/x-ndjson");
        });
    }

    private static OllamaGenerateResponseChunk BuildTerminalGenerate(string modelName, string response, DateTimeOffset start, string prompt)
    {
        long total = ServeTimings.ElapsedNanoseconds(start);
        return new OllamaGenerateResponseChunk(
            Model: modelName,
            CreatedAt: ServeTimings.UtcNow(),
            Response: response,
            Done: true,
            DoneReason: "stop",
            TotalDuration: total,
            LoadDuration: 0,
            PromptEvalCount: ServeTimings.EstimateTokens(prompt),
            PromptEvalDuration: total / 2,
            EvalCount: ServeTimings.EstimateTokens(response),
            EvalDuration: total / 2);
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
