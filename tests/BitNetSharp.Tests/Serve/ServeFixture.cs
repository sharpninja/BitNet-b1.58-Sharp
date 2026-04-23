using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.App;
using BitNetSharp.App.Serve;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Tests.Serve;

/// <summary>
/// xUnit fixture that boots the serve host on a TestServer with a stubbed
/// <see cref="IHostedAgentModel"/>. All serve tests share this fixture so the
/// host boots once per test class. The stub returns deterministic canned text
/// so route-shape assertions don't depend on model weights.
/// </summary>
public sealed class ServeFixture : IDisposable
{
    public StubHostedAgentModel Stub { get; }
    public ModelRegistry Registry { get; }
    public TestServer Server { get; }
    public HttpClient Client => Server.CreateClient();

    public ServeFixture()
    {
        Stub = new StubHostedAgentModel();
        Registry = new ModelRegistry();
        var card = ModelCard.ForHostedModel(
            Stub,
            parameterCount: 750_000_000L,
            sizeBytes: 1_392_610_228L,
            modelInfo: new Dictionary<string, object?>(StringComparer.Ordinal)
            {
                ["general.architecture"] = "bitnet",
                ["general.parameter_count"] = 750_000_000L,
                ["bitnet.context_length"] = 2048,
                ["bitnet.embedding_length"] = 768,
                ["bitnet.block_count"] = 12,
            });
        Registry.Register(Stub, card);

        var options = new ServeOptions(Host: "127.0.0.1", Port: 0, EnableCors: false);
        var builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Services.AddSingleton(Registry);
        builder.Services.AddSingleton(options);
        builder.Logging.ClearProviders();
        var app = builder.Build();

        BitNetSharp.App.Serve.Endpoints.OllamaTagsEndpoints.Map(app);
        BitNetSharp.App.Serve.Endpoints.OllamaGenerateEndpoints.Map(app);
        BitNetSharp.App.Serve.Endpoints.OllamaChatEndpoints.Map(app);
        BitNetSharp.App.Serve.Endpoints.OllamaEmbeddingsEndpoint.Map(app);
        BitNetSharp.App.Serve.Endpoints.OpenAIModelsEndpoint.Map(app);
        BitNetSharp.App.Serve.Endpoints.OpenAIChatCompletionsEndpoint.Map(app);
        BitNetSharp.App.Serve.Endpoints.OpenAICompletionsEndpoint.Map(app);

        app.StartAsync().GetAwaiter().GetResult();
        Server = app.GetTestServer();
    }

    public void Dispose()
    {
        Server.Dispose();
    }
}

public sealed class StubHostedAgentModel : IHostedAgentModel
{
    public string CannedText { get; set; } = "pong";
    public string AgentName => "stub-agent";
    public string ModelId => "bitnet-b1.58-sharp";
    public string DisplayName => "BitNet Sharp Stub";
    public string PrimaryLanguage => "en-US";
    public VerbosityLevel Verbosity => VerbosityLevel.Quiet;
    public string SystemPrompt => "You are a stub.";

    public IReadOnlyList<string> DescribeModel() => new[] { "stub" };

    public Task<HostedAgentModelResponse> GetResponseAsync(string prompt, int? maxOutputTokens = null, CancellationToken cancellationToken = default) =>
        Task.FromResult(new HostedAgentModelResponse(CannedText, Array.Empty<string>()));

    public void Dispose() { }
}
