using System;
using System.IO;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.App.Serve.Endpoints;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.App.Serve;

/// <summary>
/// Marker type for <see cref="Microsoft.AspNetCore.Mvc.Testing.WebApplicationFactory{T}"/>.
/// The test fixture targets this so it doesn't need the CLI entry point.
/// </summary>
public sealed class ServeHostMarker { }

public static class OllamaApiServer
{
    /// <summary>
    /// Starts the Ollama-compatible HTTP server and blocks until cancelled.
    /// </summary>
    public static async Task RunAsync(ModelRegistry registry, ServeOptions options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(options);

        var app = BuildApplication(registry, options);
        await app.RunAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Builds (but does not start) the <see cref="WebApplication"/>. Exposed for
    /// tests that need direct route access via <c>WebApplicationFactory</c>.
    /// </summary>
    public static WebApplication BuildApplication(ModelRegistry registry, ServeOptions options)
    {
        ArgumentNullException.ThrowIfNull(registry);
        ArgumentNullException.ThrowIfNull(options);

        var builder = WebApplication.CreateBuilder();

        builder.Logging.ClearProviders();
        builder.Logging.AddSimpleConsole(cfg =>
        {
            cfg.SingleLine = true;
            cfg.TimestampFormat = "HH:mm:ss ";
        });
        builder.Logging.SetMinimumLevel(LogLevel.Information);

        builder.WebHost.ConfigureKestrel(kestrel =>
        {
            kestrel.Limits.MaxRequestBodySize = options.MaxRequestBodyBytes;
            if (IPAddress.TryParse(options.Host, out var ip))
            {
                kestrel.Listen(ip, options.Port);
            }
            else
            {
                kestrel.ListenAnyIP(options.Port);
            }
        });

        builder.Services.AddSingleton(registry);
        builder.Services.AddSingleton(options);
        if (options.EnableCors)
        {
            builder.Services.AddCors(cors =>
            {
                cors.AddDefaultPolicy(p => p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
            });
        }
        builder.Services.AddSingleton<ServeHostMarker>();

        var app = builder.Build();

        if (options.EnableCors)
        {
            app.UseCors();
        }

        OllamaTagsEndpoints.Map(app);
        OllamaGenerateEndpoints.Map(app);
        OllamaChatEndpoints.Map(app);
        OllamaEmbeddingsEndpoint.Map(app);
        OpenAIModelsEndpoint.Map(app);
        OpenAIChatCompletionsEndpoint.Map(app);
        OpenAICompletionsEndpoint.Map(app);

        return app;
    }
}
