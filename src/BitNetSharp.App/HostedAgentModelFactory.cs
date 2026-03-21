using BitNetSharp.Core;

namespace BitNetSharp.App;

public static class HostedAgentModelFactory
{
    public const string DefaultModelId = "bitnet-b1.58-sharp";
    public const string TraditionalLocalModelId = "traditional-local";

    public static IHostedAgentModel Create(
        string? specifier,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        IEnumerable<TrainingExample>? trainingExamples = null)
    {
        var value = string.IsNullOrWhiteSpace(specifier)
            ? DefaultModelId
            : specifier.Trim();

        if (File.Exists(value))
        {
            return new LocalCommandHostedAgentModel(LocalCommandModelConfig.Load(value), verbosity);
        }

        return value.ToLowerInvariant() switch
        {
            DefaultModelId => new BitNetHostedAgentModel(
                trainingExamples is null
                    ? BitNetBootstrap.CreatePaperModel(verbosity)
                    : BitNetBootstrap.CreatePaperModel(trainingExamples, verbosity)),
            TraditionalLocalModelId => new TraditionalLocalHostedAgentModel(verbosity, trainingExamples),
            _ => throw new ArgumentException(
                $"Unknown model specifier '{value}'. Use '{DefaultModelId}', '{TraditionalLocalModelId}', or an absolute path to a local command model JSON file.",
                nameof(specifier))
        };
    }

    public static IReadOnlyList<string> BuiltInModelIds =>
    [
        DefaultModelId,
        TraditionalLocalModelId
    ];
}
