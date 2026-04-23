using BitNetSharp.App.Serve.Dto;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

internal static class OllamaEmbeddingsEndpoint
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapPost("/api/embeddings", () =>
            Results.Json(new OllamaErrorResponse("embeddings are not implemented in bitnetsharp serve v1"), ServeJson.Options, statusCode: 501));
        app.MapPost("/api/embed", () =>
            Results.Json(new OllamaErrorResponse("embeddings are not implemented in bitnetsharp serve v1"), ServeJson.Options, statusCode: 501));
    }
}
