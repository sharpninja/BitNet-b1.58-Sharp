using BitNetSharp.Core;

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
        bool enableSequenceCompression = false)
    {
        var value = string.IsNullOrWhiteSpace(specifier)
            ? DefaultModelId
            : specifier.Trim();

        if (File.Exists(value))
        {
            if (value.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            {
                return new BitNetHostedAgentModel(BitNetPaperGguf.Load(value, verbosity));
            }

            if (value.EndsWith(".bitnet.json", StringComparison.OrdinalIgnoreCase))
            {
                return new BitNetHostedAgentModel(BitNetPaperCheckpoint.Load(value, verbosity));
            }

            return new LocalCommandHostedAgentModel(LocalCommandModelConfig.Load(value), verbosity);
        }

        return value.ToLowerInvariant() switch
        {
            DefaultModelId => new BitNetHostedAgentModel(
                trainingExamples is null
                    ? BitNetBootstrap.CreatePaperModel(verbosity, enableChainBuckets, enableSequenceCompression)
                    : BitNetBootstrap.CreatePaperModel(trainingExamples, verbosity, enableChainBuckets, enableSequenceCompression)),
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
