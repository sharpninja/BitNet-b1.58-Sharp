using System.Collections.Generic;
using System.Linq;
using BitNetSharp.App.Serve.Dto;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

internal static class OllamaTagsEndpoints
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapGet("/", () => Results.Text("Ollama is running", "text/plain; charset=utf-8"));
        app.MapMethods("/", new[] { "HEAD" }, () => Results.Text("Ollama is running", "text/plain; charset=utf-8"));

        app.MapGet("/api/version", () =>
            Results.Json(new OllamaVersionResponse(ServeVersion.Current), ServeJson.Options));

        app.MapGet("/api/tags", (ModelRegistry registry) =>
        {
            var entries = registry.Enumerate().Select(r => r.Card.ToTagEntry()).ToList();
            return Results.Json(new OllamaTagListResponse(entries), ServeJson.Options);
        });

        app.MapGet("/api/ps", (ModelRegistry registry) =>
        {
            var entries = registry.Enumerate().Select(r => r.Card.ToTagEntry()).ToList();
            return Results.Json(new OllamaTagListResponse(entries), ServeJson.Options);
        });

        app.MapPost("/api/show", async (HttpRequest http, ModelRegistry registry) =>
        {
            var request = await http.ReadJsonAsync<OllamaShowRequest>();
            if (request is null || string.IsNullOrWhiteSpace(request.Model))
            {
                return Results.Json(new OllamaErrorResponse("missing or invalid 'model' field"), ServeJson.Options, statusCode: 400);
            }

            if (!registry.TryResolve(request.Model, out var entry))
            {
                return Results.Json(new OllamaErrorResponse($"model '{request.Model}' not found, try /api/tags"), ServeJson.Options, statusCode: 404);
            }

            var card = entry.Card;
            var response = new OllamaShowResponse(
                Modelfile: "# Synthetic modelfile\nFROM " + card.Name,
                Parameters: "num_ctx 2048\nnum_predict 256",
                Template: "{{ .System }}\n{{ .Prompt }}",
                Details: card.Details,
                ModelInfo: card.ModelInfo,
                Capabilities: new[] { "completion" });
            return Results.Json(response, ServeJson.Options);
        });
    }
}
