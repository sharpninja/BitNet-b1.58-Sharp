using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.App.Serve;
using BitNetSharp.App.Serve.Dto;
using BitNetSharp.Core;

namespace BitNetSharp.App;

/// <summary>
/// <c>serve</c> CLI subcommand. Hosts an Ollama-compatible HTTP API over the
/// loaded model so stock clients (Open WebUI, continue.dev, ollama-python,
/// langchain's OllamaLLM) can drive BitNetSharp without code changes.
/// </summary>
public static class ServeCommand
{
    public static async Task<int> RunAsync(string[] args, VerbosityLevel verbosity, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(args);
        var options = ParseOptions(args);

        FailFastIfPortBound(options);

        string modelSpecifier = ParseOption(args, "--model=") ?? HostedAgentModelFactory.DefaultModelId;
        var trainingExamples = BitNetTrainingCorpus.CreateDefaultExamples();

        using var hostedModel = HostedAgentModelFactory.Create(modelSpecifier, verbosity, trainingExamples);

        var registry = BuildRegistry(hostedModel);
        var card = registry.Enumerate()[0].Card;

        if (verbosity != VerbosityLevel.Quiet)
        {
            Console.WriteLine($"Serving {card.Name} on http://{options.Host}:{options.Port}");
            Console.WriteLine("Ollama-compatible routes: /api/version /api/tags /api/show /api/generate /api/chat");
            Console.WriteLine("OpenAI-compatible routes: /v1/models /v1/chat/completions /v1/completions");
        }

        await OllamaApiServer.RunAsync(registry, options, cancellationToken).ConfigureAwait(false);
        return 0;
    }

    internal static ModelRegistry BuildRegistry(IHostedAgentModel hostedModel)
    {
        ArgumentNullException.ThrowIfNull(hostedModel);

        long sizeBytes = 0;
        long parameterCount = 0;

        var modelInfo = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "bitnet",
        };

        if (hostedModel is BitNetHostedAgentModel bn)
        {
            var config = bn.Model.Config;
            sizeBytes = bn.Model.EstimateResidentParameterBytes();
            // Convert to a float32-equivalent parameter count so /api/tags'
            // parameter_size formatter ("8B", "700M", etc.) matches how
            // users think about model size.
            parameterCount = sizeBytes / 4;
            modelInfo["general.parameter_count"] = parameterCount;
            modelInfo["bitnet.context_length"] = config.MaxSequenceLength;
            modelInfo["bitnet.embedding_length"] = config.Dimension;
            modelInfo["bitnet.block_count"] = config.LayerCount;
            modelInfo["bitnet.attention.head_count"] = config.HeadCount;
            modelInfo["bitnet.attention.head_count_kv"] = config.KvHeadCount;
        }

        var card = ModelCard.ForHostedModel(hostedModel, parameterCount, sizeBytes, modelInfo);
        var registry = new ModelRegistry();
        registry.Register(hostedModel, card);
        return registry;
    }

    internal static ServeOptions ParseOptions(string[] args)
    {
        string host = ParseOption(args, "--host=") ?? "127.0.0.1";
        int port = 11434;
        string? portRaw = ParseOption(args, "--port=");
        if (!string.IsNullOrWhiteSpace(portRaw)
            && int.TryParse(portRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedPort)
            && parsedPort is > 0 and < 65536)
        {
            port = parsedPort;
        }
        bool enableCors = !args.Any(a => string.Equals(a, "--no-cors", StringComparison.OrdinalIgnoreCase));
        return new ServeOptions(Host: host, Port: port, EnableCors: enableCors);
    }

    private static void FailFastIfPortBound(ServeOptions options)
    {
        try
        {
            using var probe = new TcpListener(IPAddress.Loopback, options.Port);
            probe.Start();
            probe.Stop();
        }
        catch (SocketException ex)
        {
            throw new IOException(
                $"Port {options.Port} is already bound (is another Ollama instance running?). "
                + "Pick a free port with --port=<n>. Underlying error: " + ex.Message, ex);
        }
    }

    private static string? ParseOption(IEnumerable<string> args, string prefix) =>
        args.FirstOrDefault(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            ?.Split('=', 2)
            .LastOrDefault();
}
