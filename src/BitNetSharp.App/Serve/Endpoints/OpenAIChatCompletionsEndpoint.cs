using System;
using System.Threading.Tasks;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.App.Serve.Framing;
using BitNetSharp.App.Serve.Inference;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

internal static class OpenAIChatCompletionsEndpoint
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/chat/completions", async (HttpContext http, ModelRegistry registry) =>
        {
            var request = await http.Request.ReadJsonAsync<OpenAIChatCompletionRequest>(http.RequestAborted);
            if (request is null || string.IsNullOrWhiteSpace(request.Model))
            {
                return Results.Json(BuildError("missing or invalid 'model' field", "invalid_request_error", "invalid_request", "model"), ServeJson.Options, statusCode: 400);
            }
            if (request.Messages is null || request.Messages.Count == 0)
            {
                return Results.Json(BuildError("'messages' must be a non-empty array", "invalid_request_error", "invalid_request", "messages"), ServeJson.Options, statusCode: 400);
            }
            if (!registry.TryResolve(request.Model, out var entry))
            {
                return Results.Json(BuildError($"model '{request.Model}' not found", "invalid_request_error", "model_not_found", "model"), ServeJson.Options, statusCode: 404);
            }

            bool stream = request.Stream ?? false;
            string prompt = ChatPromptAssembler.Assemble(entry.Model.SystemPrompt, request.Messages);
            var response = await entry.Model.GetResponseAsync(prompt, request.MaxTokens, http.RequestAborted).ConfigureAwait(false);
            long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            string id = ServeTimings.NewChatCompletionId();

            if (!stream)
            {
                var choice = new OpenAIChoice(
                    Index: 0,
                    Message: new OpenAIChatMessage("assistant", response.Text),
                    FinishReason: "stop");
                int promptTokens = ServeTimings.EstimateTokens(prompt);
                int completionTokens = ServeTimings.EstimateTokens(response.Text);
                var payload = new OpenAIChatCompletionResponse(
                    Id: id,
                    Object: "chat.completion",
                    Created: created,
                    Model: entry.Card.Name,
                    Choices: new[] { choice },
                    Usage: new OpenAIUsage(promptTokens, completionTokens, promptTokens + completionTokens));
                return Results.Json(payload, ServeJson.Options);
            }

            return Results.Stream(async (stream) =>
            {
                var writer = new SseWriter(stream);
                if (!string.IsNullOrEmpty(response.Text))
                {
                    var deltaChunk = new OpenAIChatCompletionChunk(
                        Id: id,
                        Object: "chat.completion.chunk",
                        Created: created,
                        Model: entry.Card.Name,
                        Choices: new[]
                        {
                            new OpenAIChunkChoice(
                                Index: 0,
                                Delta: new OpenAIChatDelta(Role: "assistant", Content: response.Text),
                                FinishReason: null),
                        });
                    await writer.WriteAsync(deltaChunk, http.RequestAborted).ConfigureAwait(false);
                }

                var terminal = new OpenAIChatCompletionChunk(
                    Id: id,
                    Object: "chat.completion.chunk",
                    Created: created,
                    Model: entry.Card.Name,
                    Choices: new[]
                    {
                        new OpenAIChunkChoice(
                            Index: 0,
                            Delta: new OpenAIChatDelta(Role: null, Content: null),
                            FinishReason: "stop"),
                    });
                await writer.WriteAsync(terminal, http.RequestAborted).ConfigureAwait(false);
                await writer.WriteDoneAsync(http.RequestAborted).ConfigureAwait(false);
            }, contentType: "text/event-stream");
        });
    }

    internal static OpenAIErrorEnvelope BuildError(string message, string type, string? code, string? param) =>
        new(new OpenAIErrorBody(message, type, code, param));
}
