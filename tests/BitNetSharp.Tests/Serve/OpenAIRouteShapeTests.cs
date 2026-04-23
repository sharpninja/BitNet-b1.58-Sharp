using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;

namespace BitNetSharp.Tests.Serve;

public sealed class OpenAIRouteShapeTests : IClassFixture<ServeFixture>
{
    private readonly ServeFixture _fixture;
    public OpenAIRouteShapeTests(ServeFixture fixture) => _fixture = fixture;

    [Fact]
    public async Task GetV1Models_ReturnsOpenAIList()
    {
        var client = _fixture.Client;
        var doc = JsonDocument.Parse(await client.GetStringAsync("/v1/models"));
        Assert.Equal("list", doc.RootElement.GetProperty("object").GetString());
        var data = doc.RootElement.GetProperty("data");
        Assert.Equal(1, data.GetArrayLength());
        var first = data[0];
        Assert.Equal("bitnet-b1.58-sharp", first.GetProperty("id").GetString());
        Assert.Equal("model", first.GetProperty("object").GetString());
        Assert.True(first.GetProperty("created").GetInt64() > 0);
        Assert.Equal("bitnetsharp", first.GetProperty("owned_by").GetString());
    }

    [Fact]
    public async Task PostV1ChatCompletions_NonStreaming_ReturnsChoicesArray()
    {
        _fixture.Stub.CannedText = "oa-reply";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/v1/chat/completions", new
        {
            model = "bitnet-b1.58-sharp",
            messages = new[] { new { role = "user", content = "hi" } },
            stream = false,
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("chat.completion", doc.RootElement.GetProperty("object").GetString());
        Assert.StartsWith("chatcmpl-", doc.RootElement.GetProperty("id").GetString());
        var choices = doc.RootElement.GetProperty("choices");
        Assert.Equal(1, choices.GetArrayLength());
        var choice = choices[0];
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
        var message = choice.GetProperty("message");
        Assert.Equal("assistant", message.GetProperty("role").GetString());
        Assert.Equal("oa-reply", message.GetProperty("content").GetString());
        Assert.True(doc.RootElement.GetProperty("usage").GetProperty("total_tokens").GetInt32() > 0);
    }

    [Fact]
    public async Task PostV1ChatCompletions_Streaming_EmitsSseWithDoneTerminator()
    {
        _fixture.Stub.CannedText = "oa-stream";
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
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType!.MediaType);
        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("data: ", body);
        Assert.EndsWith("data: [DONE]\n\n", body);
        var frames = body.Split("\n\n", System.StringSplitOptions.RemoveEmptyEntries);
        // Expect at least: content delta, terminal (finish_reason=stop), [DONE]
        Assert.True(frames.Length >= 3);
        var firstJson = frames[0].Substring("data: ".Length);
        var firstDoc = JsonDocument.Parse(firstJson);
        Assert.Equal("chat.completion.chunk", firstDoc.RootElement.GetProperty("object").GetString());
    }

    [Fact]
    public async Task PostV1Completions_Legacy_ReturnsTextChoice()
    {
        _fixture.Stub.CannedText = "oa-legacy";
        var client = _fixture.Client;
        var response = await client.PostAsJsonAsync("/v1/completions", new
        {
            model = "bitnet-b1.58-sharp",
            prompt = "hi",
        });
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("text_completion", doc.RootElement.GetProperty("object").GetString());
        var choice = doc.RootElement.GetProperty("choices")[0];
        Assert.Equal("oa-legacy", choice.GetProperty("text").GetString());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
    }
}
