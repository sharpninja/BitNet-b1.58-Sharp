using BitNetSharp.Core;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.App;

public static class HostedAgentModelFactory
{
    public const string DefaultModelId = "bitnet-b1.58-sharp";
    public const string TraditionalLocalModelId = "traditional-local";

    public static IHostedAgentModel Create(
        string? specifier,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        IEnumerable<TrainingExample>? trainingExamples = null,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false,
        ILoggerFactory? loggerFactory = null)
    {
        var value = string.IsNullOrWhiteSpace(specifier)
            ? DefaultModelId
            : specifier.Trim();
        var lf = loggerFactory ?? NullLoggerFactory.Instance;

        if (File.Exists(value))
        {
            if (value.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                return new BitNetHostedAgentModel(BitNetPaperGguf.Load(value, lf.CreateLogger<BitNetPaperModel>(), lf, verbosity));
            }

            if (value.EndsWith(".bitnet.json", StringComparison.OrdinalIgnoreCase))
            {
                return new BitNetHostedAgentModel(BitNetPaperCheckpoint.Load(value, lf.CreateLogger<BitNetPaperModel>(), lf, verbosity));
            }

            return new LocalCommandHostedAgentModel(LocalCommandModelConfig.Load(value), verbosity);
        }

        return value.ToLowerInvariant() switch
        {
            DefaultModelId => new BitNetHostedAgentModel(
                trainingExamples is null
                    ? BitNetBootstrap.CreatePaperModel(verbosity, enableChainBuckets, enableSequenceCompression, lf)
                    : BitNetBootstrap.CreatePaperModel(trainingExamples, verbosity, enableChainBuckets, enableSequenceCompression, lf)),
            TraditionalLocalModelId => new TraditionalLocalHostedAgentModel(verbosity, trainingExamples),
            _ => throw new ArgumentException(
                $"Unknown model specifier '{value}'. Use '{DefaultModelId}', '{TraditionalLocalModelId}', or an absolute path to a repo-authored .bitnet.json/.gguf model or local command model JSON file.",
                nameof(specifier))
        };
    }

    public static IReadOnlyList<string> BuiltInModelIds =>
    [
        DefaultModelId,
        TraditionalLocalModelId
    ];
}
