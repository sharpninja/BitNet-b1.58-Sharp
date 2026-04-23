using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using BitNetSharp.App.Serve;
using BitNetSharp.App.Serve.Dto;

namespace BitNetSharp.Tests.Serve;

public sealed class OllamaRouteShapeTests : IClassFixture<ServeFixture>
{
    private readonly ServeFixture _fixture;
    public OllamaRouteShapeTests(ServeFixture fixture) => _fixture = fixture;

    [Fact]
    public async Task GetApiVersion_ReturnsVersionEnvelope()
    {
        var client = _fixture.Client;
        var doc = JsonDocument.Parse(await client.GetStringAsync("/api/version"));
        Assert.True(doc.RootElement.TryGetProperty("version", out var v));
        Assert.False(string.IsNullOrWhiteSpace(v.GetString()));
    }

    [Fact]
    public async Task GetApiTags_ReturnsModelListWithRequiredFields()
    {
        var client = _fixture.Client;
        var doc = JsonDocument.Parse(await client.GetStringAsync("/api/tags"));
        var models = doc.RootElement.GetProperty("models");
        Assert.Equal(1, models.GetArrayLength());
        var first = models[0];
        Assert.Equal("bitnet-b1.58-sharp:latest", first.GetProperty("name").GetString());
        Assert.Equal("bitnet-b1.58-sharp:latest", first.GetProperty("model").GetString());
        Assert.False(string.IsNullOrWhiteSpace(first.GetProperty("modified_at").GetString()));
        Assert.True(first.GetProperty("size").GetInt64() > 0);
        Assert.StartsWith("sha256:", first.GetProperty("digest").GetString());
        var details = first.GetProperty("details");
        Assert.Equal("gguf", details.GetProperty("format").GetString());
        Assert.Equal("bitnet", details.GetProperty("family").GetString());
        Assert.Equal("b1.58", details.GetProperty("quantization_level").GetString());
    }

    [Fact]
    public async Task PostApiShow_ReturnsModelMetadata()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/show", new { model = "bitnet-b1.58-sharp" });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.True(doc.RootElement.TryGetProperty("modelfile", out _));
        Assert.True(doc.RootElement.TryGetProperty("parameters", out _));
        Assert.True(doc.RootElement.TryGetProperty("template", out _));
        Assert.True(doc.RootElement.TryGetProperty("details", out _));
        Assert.True(doc.RootElement.TryGetProperty("model_info", out _));
        var capabilities = doc.RootElement.GetProperty("capabilities");
        Assert.Equal(1, capabilities.GetArrayLength());
        Assert.Equal("completion", capabilities[0].GetString());
    }

    [Fact]
    public async Task PostApiChat_NonStreaming_WrapsMessageWithDone()
    {
        _fixture.Stub.CannedText = "hello-from-stub";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = "bitnet-b1.58-sharp",
            messages = new[] { new { role = "user", content = "ping" } },
            stream = false,
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.True(doc.RootElement.GetProperty("done").GetBoolean());
        Assert.Equal("stop", doc.RootElement.GetProperty("done_reason").GetString());
        var message = doc.RootElement.GetProperty("message");
        Assert.Equal("assistant", message.GetProperty("role").GetString());
        Assert.Equal("hello-from-stub", message.GetProperty("content").GetString());
    }

    [Fact]
    public async Task PostApiChat_Streaming_EmitsNdjsonWithDoneTerminator()
    {
        _fixture.Stub.CannedText = "streamed";
        var client = _fixture.Client;
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = JsonContent.Create(new
            {
                model = "bitnet-b1.58-sharp",
                messages = new[] { new { role = "user", content = "ping" } },
                stream = true,
            }),
        };
        var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/x-ndjson", response.Content.Headers.ContentType!.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        var lines = body.Split('\n', System.StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(2, lines.Length);
        var first = JsonDocument.Parse(lines[0]).RootElement;
        Assert.False(first.GetProperty("done").GetBoolean());
        Assert.Equal("streamed", first.GetProperty("message").GetProperty("content").GetString());
        var second = JsonDocument.Parse(lines[1]).RootElement;
        Assert.True(second.GetProperty("done").GetBoolean());
        Assert.Equal("stop", second.GetProperty("done_reason").GetString());
    }

    [Fact]
    public async Task PostApiGenerate_NonStreaming_ReturnsSingleObject()
    {
        _fixture.Stub.CannedText = "generated";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/generate", new
        {
            model = "bitnet-b1.58-sharp",
            prompt = "hi",
            stream = false,
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("generated", doc.RootElement.GetProperty("response").GetString());
        Assert.True(doc.RootElement.GetProperty("done").GetBoolean());
    }

    [Fact]
    public async Task PostApiGenerate_Streaming_EmitsNdjsonWithDoneTerminator()
    {
        _fixture.Stub.CannedText = "gen-stream";
        var client = _fixture.Client;
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/generate")
        {
            Content = JsonContent.Create(new
            {
                model = "bitnet-b1.58-sharp",
                prompt = "hi",
                stream = true,
            }),
        };
        var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("application/x-ndjson", response.Content.Headers.ContentType!.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        var lines = body.Split('\n', System.StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(2, lines.Length);
        Assert.Equal("gen-stream", JsonDocument.Parse(lines[0]).RootElement.GetProperty("response").GetString());
        Assert.True(JsonDocument.Parse(lines[1]).RootElement.GetProperty("done").GetBoolean());
    }

    [Fact]
    public async Task PostApiEmbeddings_Returns501()
    {
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/embeddings", new { model = "bitnet-b1.58-sharp", prompt = "x" });
        Assert.Equal(HttpStatusCode.NotImplemented, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Contains("embeddings", doc.RootElement.GetProperty("error").GetString());
    }

    // Ollama wire allows keep_alive as either a duration string ("5m", "10s",
    // "-1", "0") or an integer number of seconds. Real ollama CLI sends a
    // number; open-webui sends a string. We accept and ignore both rather
    // than 500ing on type mismatch.
    [Fact]
    public async Task PostApiChat_KeepAliveAsNumber_Accepted()
    {
        _fixture.Stub.CannedText = "ok";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = "bitnet-b1.58-sharp",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
            keep_alive = 300,
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task PostApiChat_KeepAliveAsString_Accepted()
    {
        _fixture.Stub.CannedText = "ok";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = "bitnet-b1.58-sharp",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
            keep_alive = "5m",
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task PostApiGenerate_KeepAliveAsNumber_Accepted()
    {
        _fixture.Stub.CannedText = "ok";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/api/generate", new
        {
            model = "bitnet-b1.58-sharp",
            prompt = "hi",
            stream = false,
            keep_alive = 0,
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }
}
