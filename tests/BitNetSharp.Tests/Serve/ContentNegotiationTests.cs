using System.Net.Http;
using System.Net.Http.Json;

namespace BitNetSharp.Tests.Serve;

public sealed class ContentNegotiationTests : IClassFixture<ServeFixture>
{
    private readonly ServeFixture _fixture;
    public ContentNegotiationTests(ServeFixture fixture) => _fixture = fixture;

    [Fact]
    public async Task V1ChatCompletions_StreamTrue_UsesEventStream()
    {
        var client = _fixture.Client;
        var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
        {
            Content = JsonContent.Create(new
            {
                model = "bitnet-b1.58-sharp",
                messages = new[] { new { role = "user", content = "hi" } },
                stream = true,
            }),
        };
        var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType!.MediaType);
    }

    [Fact]
    public async Task V1ChatCompletions_StreamFalse_UsesApplicationJson()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "bitnet-b1.58-sharp",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });
        Assert.Equal("application/json", response.Content.Headers.ContentType!.MediaType);
    }

    [Fact]
    public async Task ApiChat_StreamDefaultTrue_UsesNdjson()
    {
        var client = _fixture.Client;
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = JsonContent.Create(new
            {
                model = "bitnet-b1.58-sharp",
                messages = new[] { new { role = "user", content = "hi" } },
                // no "stream" key: Ollama default is true
            }),
        };
        var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("application/x-ndjson", response.Content.Headers.ContentType!.MediaType);
    }

    [Fact]
    public async Task ApiChat_StreamFalse_UsesApplicationJson()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = "bitnet-b1.58-sharp",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });
        Assert.Equal("application/json", response.Content.Headers.ContentType!.MediaType);
    }
}
