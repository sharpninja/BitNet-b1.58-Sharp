using System;
using System.Threading.Tasks;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.App.Serve.Inference;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

/// <summary>
/// Legacy OpenAI completions endpoint. Maps the incoming <c>prompt</c>
/// straight to the hosted model without applying a chat template (per
/// plan decision 6.3 option A for /v1/chat/completions, but legacy
/// completions is explicitly raw).
/// </summary>
internal static class OpenAICompletionsEndpoint
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapPost("/v1/completions", async (HttpContext http, ModelRegistry registry) =>
        {
            var request = await http.Request.ReadJsonAsync<OpenAICompletionRequest>(http.RequestAborted);
            if (request is null || string.IsNullOrWhiteSpace(request.Model))
            {
                return Results.Json(OpenAIChatCompletionsEndpoint.BuildError("missing or invalid 'model' field", "invalid_request_error", "invalid_request", "model"), ServeJson.Options, statusCode: 400);
            }
            if (!registry.TryResolve(request.Model, out var entry))
            {
                return Results.Json(OpenAIChatCompletionsEndpoint.BuildError($"model '{request.Model}' not found", "invalid_request_error", "model_not_found", "model"), ServeJson.Options, statusCode: 404);
            }

            var response = await entry.Model.GetResponseAsync(request.Prompt ?? string.Empty, request.MaxTokens, http.RequestAborted).ConfigureAwait(false);
            long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            string id = ServeTimings.NewChatCompletionId().Replace("chatcmpl-", "cmpl-", StringComparison.Ordinal);

            var choice = new OpenAICompletionChoice(Index: 0, Text: response.Text, FinishReason: "stop");
            int promptTokens = ServeTimings.EstimateTokens(request.Prompt);
            int completionTokens = ServeTimings.EstimateTokens(response.Text);
            var payload = new OpenAICompletionResponse(
                Id: id,
                Object: "text_completion",
                Created: created,
                Model: entry.Card.Name,
                Choices: new[] { choice },
                Usage: new OpenAIUsage(promptTokens, completionTokens, promptTokens + completionTokens));
            return Results.Json(payload, ServeJson.Options);
        });
    }
}
