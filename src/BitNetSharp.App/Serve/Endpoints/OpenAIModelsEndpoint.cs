using System.Linq;
using BitNetSharp.App.Serve.Dto;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace BitNetSharp.App.Serve.Endpoints;

internal static class OpenAIModelsEndpoint
{
    public static void Map(IEndpointRouteBuilder app)
    {
        app.MapGet("/v1/models", (ModelRegistry registry) =>
        {
            var data = registry.Enumerate().Select(r => r.Card.ToOpenAIEntry()).ToList();
            return Results.Json(new OpenAIModelList("list", data), ServeJson.Options);
        });
    }
}
