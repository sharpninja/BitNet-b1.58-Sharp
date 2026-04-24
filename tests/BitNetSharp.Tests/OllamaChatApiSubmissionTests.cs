using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
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
/// Exercises the Ollama-compatible /api/chat endpoint through an in-memory
/// TestServer (no real TCP bind). Parameterizes both the client query and
/// the server-side model data so future performance changes can be regressed
/// against a deterministic stub instead of a real 348M-param network.
/// </summary>
public sealed class OllamaChatApiSubmissionTests
{
    [Theory]
    [InlineData("alpha", "You are alpha.", "hello", 2, "alpha-answers")]
    [InlineData("beta:latest", "You are beta.", "how are you", 4, "beta-answers")]
    [InlineData("tiny-b1.58", "strict system", "single", 1, "ok")]
    public async Task Chat_Non_Streaming_Returns_Parameterized_Model_And_Response(
        string clientRequestedModel,
        string systemPrompt,
        string userQuery,
        int numPredict,
        string cannedResponse)
    {
        string baseModelId = clientRequestedModel.Contains(':', StringComparison.Ordinal)
            ? clientRequestedModel[..clientRequestedModel.IndexOf(':', StringComparison.Ordinal)]
            : clientRequestedModel;

        var stubModel = new StubHostedAgentModel(
            modelId: baseModelId,
            systemPrompt: systemPrompt,
            cannedResponse: cannedResponse);

        using var host = await BuildTestHostAsync(stubModel);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: clientRequestedModel,
            Messages: new[] { new OllamaChatMessage("user", userQuery) },
            Stream: false,
            Options: new Dictionary<string, object?> { ["num_predict"] = numPredict });

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var chunk = await response.Content.ReadFromJsonAsync<OllamaChatResponseChunk>(ServeJson.Options);
        Assert.NotNull(chunk);
        Assert.True(chunk!.Done);
        Assert.Equal("stop", chunk.DoneReason);
        Assert.Equal(baseModelId, chunk.Model);
        Assert.NotNull(chunk.Message);
        Assert.Equal("assistant", chunk.Message.Role);
        Assert.Equal(cannedResponse, chunk.Message.Content);
        Assert.Equal(numPredict, stubModel.LastMaxOutputTokens);
        Assert.Contains(userQuery, stubModel.LastPrompt, StringComparison.Ordinal);
        Assert.Contains(systemPrompt, stubModel.LastPrompt, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("alpha", "msg one", 1)]
    [InlineData("beta", "msg two", 3)]
    public async Task Chat_Streaming_Emits_Terminal_Chunk_With_Done_True(
        string modelId,
        string userQuery,
        int numPredict)
    {
        var stubModel = new StubHostedAgentModel(modelId, "sys", "streamed");

        using var host = await BuildTestHostAsync(stubModel);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: modelId,
            Messages: new[] { new OllamaChatMessage("user", userQuery) },
            Stream: true,
            Options: new Dictionary<string, object?> { ["num_predict"] = numPredict });

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        response.EnsureSuccessStatusCode();

        Assert.Equal("application/x-ndjson", response.Content.Headers.ContentType?.MediaType);

        var body = await response.Content.ReadAsStringAsync();
        string[] lines = body.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.NotEmpty(lines);

        var terminal = JsonSerializer.Deserialize<OllamaChatResponseChunk>(lines[^1], ServeJson.Options);
        Assert.NotNull(terminal);
        Assert.True(terminal!.Done);
        Assert.Equal("stop", terminal.DoneReason);
    }

    [Fact]
    public async Task Chat_Returns_404_For_Unknown_Model()
    {
        var stubModel = new StubHostedAgentModel("known", "sys", "hi");
        using var host = await BuildTestHostAsync(stubModel);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: "unknown",
            Messages: new[] { new OllamaChatMessage("user", "hi") },
            Stream: false);

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task Chat_Returns_400_For_Empty_Messages()
    {
        var stubModel = new StubHostedAgentModel("known", "sys", "hi");
        using var host = await BuildTestHostAsync(stubModel);
        var client = host.GetTestClient();

        var chatRequest = new OllamaChatRequest(
            Model: "known",
            Messages: Array.Empty<OllamaChatMessage>(),
            Stream: false);

        using var response = await client.PostAsJsonAsync("/api/chat", chatRequest, ServeJson.Options);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
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

    /// <summary>
    /// Deterministic IHostedAgentModel stub. Captures the last prompt and max
    /// output tokens so tests can verify ChatPromptAssembler + endpoint wiring
    /// without needing a real neural network.
    /// </summary>
    private sealed class StubHostedAgentModel : IHostedAgentModel
    {
        private readonly string _cannedResponse;

        public StubHostedAgentModel(string modelId, string systemPrompt, string cannedResponse)
        {
            ModelId = modelId;
            AgentName = modelId;
            SystemPrompt = systemPrompt;
            _cannedResponse = cannedResponse;
        }

        public string AgentName { get; }
        public string ModelId { get; }
        public string DisplayName => "Stub hosted agent";
        public string PrimaryLanguage => "en";
        public VerbosityLevel Verbosity => VerbosityLevel.Quiet;
        public string SystemPrompt { get; }

        public string? LastPrompt { get; private set; }
        public int? LastMaxOutputTokens { get; private set; }

        public IReadOnlyList<string> DescribeModel() => new[] { DisplayName };

        public Task<HostedAgentModelResponse> GetResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            LastPrompt = prompt;
            LastMaxOutputTokens = maxOutputTokens;
            return Task.FromResult(new HostedAgentModelResponse(_cannedResponse, Array.Empty<string>()));
        }

        public void Dispose()
        {
        }
    }
}
