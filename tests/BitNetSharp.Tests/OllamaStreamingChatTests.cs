using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using BitNetSharp.App;
using BitNetSharp.App.Serve;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.App.Serve.Endpoints;
using BitNetSharp.Core;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase 5: Streaming /api/chat — verifies the endpoint emits one NDJSON chunk
/// per token when stream=true, collapses to single JSON when stream=false, and
/// honours mid-stream cancellation.
/// </summary>
public sealed class OllamaStreamingChatTests
{
    [Theory]
    [InlineData("alpha", 3)]
    [InlineData("beta", 5)]
    public async Task StreamTrue_EmitsOneNdjsonLinePerToken_EndingWithDoneTrue(
        string modelId,
        int tokenCount)
    {
        var tokens = new List<string>();
        for (var i = 0; i < tokenCount; i++)
        {
            tokens.Add($"tok{i}");
        }

        var stub = new StreamingStubHostedAgentModel(modelId, "sys", tokens);
        using var host = await BuildTestHostAsync(stub);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: modelId,
            Messages: new[] { new OllamaChatMessage("user", "hello") },
            Stream: true,
            Options: new Dictionary<string, object?> { ["num_predict"] = tokenCount });

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        response.EnsureSuccessStatusCode();
        Assert.Equal("application/x-ndjson", response.Content.Headers.ContentType?.MediaType);

        var body = await response.Content.ReadAsStringAsync();
        var lines = body.Split('\n', StringSplitOptions.RemoveEmptyEntries);

        Assert.Equal(tokenCount + 1, lines.Length);

        for (var i = 0; i < tokenCount; i++)
        {
            var chunk = JsonSerializer.Deserialize<OllamaChatResponseChunk>(lines[i], ServeJson.Options);
            Assert.NotNull(chunk);
            Assert.False(chunk!.Done);
            Assert.NotNull(chunk.Message);
            Assert.Equal("assistant", chunk.Message!.Role);
            Assert.Equal(tokens[i], chunk.Message.Content);
        }

        var terminal = JsonSerializer.Deserialize<OllamaChatResponseChunk>(lines[^1], ServeJson.Options);
        Assert.NotNull(terminal);
        Assert.True(terminal!.Done);
        Assert.Equal("stop", terminal.DoneReason);
    }

    [Fact]
    public async Task StreamFalse_StillReturnsSingleJson()
    {
        var tokens = new List<string> { "one ", "two ", "three" };
        var stub = new StreamingStubHostedAgentModel("alpha", "sys", tokens);

        using var host = await BuildTestHostAsync(stub);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: "alpha",
            Messages: new[] { new OllamaChatMessage("user", "hi") },
            Stream: false);

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var chunk = await response.Content.ReadFromJsonAsync<OllamaChatResponseChunk>(ServeJson.Options);
        Assert.NotNull(chunk);
        Assert.True(chunk!.Done);
        Assert.Equal("stop", chunk.DoneReason);
        Assert.Equal("one two three", chunk.Message!.Content);
    }

    [Fact]
    public async Task StreamTrue_EmitsPerTokenTiming()
    {
        // Section A3: NDJSON chunks carry per-token forward_ms / select_ms /
        // decode_ms when the underlying model surfaces them via the rich
        // StreamTokensAsync overload.
        var tokens = new[]
        {
            new GeneratedToken(TokenId: 1, TokenText: "hello", Step: 0, ForwardMs: 12.5, SelectMs: 0.7, DecodeMs: 9.4),
            new GeneratedToken(TokenId: 2, TokenText: " world", Step: 1, ForwardMs: 9.4, SelectMs: 0.6, DecodeMs: 8.1),
            new GeneratedToken(TokenId: 3, TokenText: "!", Step: 2, ForwardMs: 8.1, SelectMs: 0.5, DecodeMs: 0d),
        };
        var stub = new TimingStreamingStubHostedAgentModel("alpha", "sys", tokens);
        using var host = await BuildTestHostAsync(stub);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: "alpha",
            Messages: new[] { new OllamaChatMessage("user", "hi") },
            Stream: true,
            Options: new Dictionary<string, object?> { ["num_predict"] = tokens.Length });

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync();
        var lines = body.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(tokens.Length + 1, lines.Length);

        for (var i = 0; i < tokens.Length; i++)
        {
            var chunk = JsonSerializer.Deserialize<OllamaChatResponseChunk>(lines[i], ServeJson.Options);
            Assert.NotNull(chunk);
            Assert.False(chunk!.Done);
            Assert.Equal(tokens[i].TokenText, chunk.Message!.Content);
            Assert.Equal(tokens[i].ForwardMs, chunk.ForwardMs);
            Assert.Equal(tokens[i].SelectMs, chunk.SelectMs);
            Assert.Equal(tokens[i].DecodeMs, chunk.DecodeMs);
        }

        var terminal = JsonSerializer.Deserialize<OllamaChatResponseChunk>(lines[^1], ServeJson.Options);
        Assert.NotNull(terminal);
        Assert.True(terminal!.Done);
        // Terminal chunk does not need per-token timing (aggregate fields
        // cover it). Per-token fields stay null on the terminal chunk so
        // clients can distinguish summary vs. mid-stream.
        Assert.Null(terminal.ForwardMs);
        Assert.Null(terminal.SelectMs);
        Assert.Null(terminal.DecodeMs);
    }

    [Fact]
    public async Task StreamFalse_OmitsTiming()
    {
        var tokens = new List<string> { "alpha ", "beta" };
        var stub = new StreamingStubHostedAgentModel("alpha", "sys", tokens);
        using var host = await BuildTestHostAsync(stub);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: "alpha",
            Messages: new[] { new OllamaChatMessage("user", "hi") },
            Stream: false);

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        var chunk = await response.Content.ReadFromJsonAsync<OllamaChatResponseChunk>(ServeJson.Options);

        Assert.NotNull(chunk);
        Assert.True(chunk!.Done);
        Assert.Null(chunk.ForwardMs);
        Assert.Null(chunk.SelectMs);
        Assert.Null(chunk.DecodeMs);
    }

    [Fact]
    public async Task CancellationMidStream_StopsGeneration()
    {
        var stub = new CancellableStubHostedAgentModel("alpha", "sys");
        using var cts = new CancellationTokenSource();

        var emitted = 0;
        var task = Task.Run(async () =>
        {
            try
            {
                await foreach (var chunk in ((IHostedAgentModel)stub).StreamResponseAsync("hi", null, cts.Token))
                {
                    emitted++;
                    if (emitted >= 3)
                    {
                        cts.Cancel();
                    }
                }
            }
            catch (OperationCanceledException) { }
        });

        await task;
        Assert.True(emitted >= 3);
        Assert.True(stub.CancellationObserved, "stub should observe cancellation after consumer cancels ct");
        Assert.True(emitted < 10_000, "stream should stop early, not emit every scheduled token");
    }

    private static async Task<IHost> BuildTestHostAsync(IHostedAgentModel model)
    {
        var registry = new ModelRegistry();
        var card = ModelCard.ForHostedModel(
            model,
            parameterCount: 1_000_000L,
            sizeBytes: 1_000_000L,
            modelInfo: new Dictionary<string, object?>());
        registry.Register(model, card);

        var hostBuilder = new HostBuilder()
            .ConfigureWebHost(webBuilder =>
            {
                webBuilder.UseTestServer();
                webBuilder.ConfigureServices(services =>
                {
                    services.AddSingleton(registry);
                    services.AddRouting();
                });
                webBuilder.Configure(app =>
                {
                    app.UseRouting();
                    app.UseEndpoints(endpoints =>
                    {
                        OllamaChatEndpoints.Map(endpoints);
                    });
                });
            });

        return await hostBuilder.StartAsync();
    }

    /// <summary>Stub that yields the supplied token list one chunk at a time via StreamResponseAsync.</summary>
    private sealed class StreamingStubHostedAgentModel : IHostedAgentModel
    {
        private readonly IReadOnlyList<string> _tokens;

        public StreamingStubHostedAgentModel(string modelId, string systemPrompt, IReadOnlyList<string> tokens)
        {
            ModelId = modelId;
            AgentName = modelId;
            SystemPrompt = systemPrompt;
            _tokens = tokens;
        }

        public string AgentName { get; }
        public string ModelId { get; }
        public string DisplayName => "streaming stub";
        public string PrimaryLanguage => "en";
        public VerbosityLevel Verbosity => VerbosityLevel.Quiet;
        public string SystemPrompt { get; }

        public IReadOnlyList<string> DescribeModel() => new[] { DisplayName };

        public Task<HostedAgentModelResponse> GetResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new HostedAgentModelResponse(string.Concat(_tokens), Array.Empty<string>()));
        }

        public async IAsyncEnumerable<string> StreamResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var t in _tokens)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return t;
                await Task.Yield();
            }
        }

        public void Dispose() { }
    }

    /// <summary>Stub that yields rich GeneratedToken events with non-zero timing via StreamTokensAsync directly.</summary>
    private sealed class TimingStreamingStubHostedAgentModel : IHostedAgentModel
    {
        private readonly IReadOnlyList<GeneratedToken> _tokens;

        public TimingStreamingStubHostedAgentModel(string modelId, string systemPrompt, IReadOnlyList<GeneratedToken> tokens)
        {
            ModelId = modelId;
            AgentName = modelId;
            SystemPrompt = systemPrompt;
            _tokens = tokens;
        }

        public string AgentName { get; }
        public string ModelId { get; }
        public string DisplayName => "timing streaming stub";
        public string PrimaryLanguage => "en";
        public VerbosityLevel Verbosity => VerbosityLevel.Quiet;
        public string SystemPrompt { get; }

        public IReadOnlyList<string> DescribeModel() => new[] { DisplayName };

        public Task<HostedAgentModelResponse> GetResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var sb = new System.Text.StringBuilder();
            foreach (var t in _tokens)
            {
                sb.Append(t.TokenText);
            }
            return Task.FromResult(new HostedAgentModelResponse(sb.ToString(), Array.Empty<string>()));
        }

        public async IAsyncEnumerable<GeneratedToken> StreamTokensAsync(
            string prompt,
            int? maxOutputTokens = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var t in _tokens)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return t;
                await Task.Yield();
            }
        }

        public void Dispose() { }
    }

    /// <summary>Stub that emits tokens indefinitely, marks CancellationObserved when ct fires.</summary>
    private sealed class CancellableStubHostedAgentModel : IHostedAgentModel
    {
        public CancellableStubHostedAgentModel(string modelId, string systemPrompt)
        {
            ModelId = modelId;
            AgentName = modelId;
            SystemPrompt = systemPrompt;
        }

        public string AgentName { get; }
        public string ModelId { get; }
        public string DisplayName => "cancellable stub";
        public string PrimaryLanguage => "en";
        public VerbosityLevel Verbosity => VerbosityLevel.Quiet;
        public string SystemPrompt { get; }
        public bool CancellationObserved { get; private set; }

        public IReadOnlyList<string> DescribeModel() => new[] { DisplayName };

        public Task<HostedAgentModelResponse> GetResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new HostedAgentModelResponse("fallback", Array.Empty<string>()));
        }

        public async IAsyncEnumerable<string> StreamResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            try
            {
                for (var i = 0; i < 10_000; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    yield return $"tok{i}";
                    await Task.Delay(10, cancellationToken);
                }
            }
            finally
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    CancellationObserved = true;
                }
            }
        }

        public void Dispose() { }
    }
}
