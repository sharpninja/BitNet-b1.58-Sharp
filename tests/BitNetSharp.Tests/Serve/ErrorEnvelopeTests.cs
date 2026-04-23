using System.Net;
using System.Net.Http.Json;
using System.Text.Json;

namespace BitNetSharp.Tests.Serve;

public sealed class ErrorEnvelopeTests : IClassFixture<ServeFixture>
{
    private readonly ServeFixture _fixture;
    public ErrorEnvelopeTests(ServeFixture fixture) => _fixture = fixture;

    [Fact]
    public async Task ApiChat_UnknownModel_Returns404WithOllamaShape()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = "unknown-model",
            messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var error = doc.RootElement.GetProperty("error").GetString();
        Assert.Contains("unknown-model", error);
    }

    [Fact]
    public async Task V1ChatCompletions_UnknownModel_Returns404WithOpenAIWrappedShape()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "unknown-model",
            messages = new[] { new { role = "user", content = "hi" } },
        });
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("model_not_found", error.GetProperty("code").GetString());
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("model", error.GetProperty("param").GetString());
    }

    [Fact]
    public async Task ApiChat_EmptyMessages_Returns400()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = "bitnet-b1.58-sharp",
            messages = System.Array.Empty<object>(),
        });
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task ApiGenerate_MissingModel_Returns400()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/generate", new
        {
            prompt = "hi",
        });
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }
}
