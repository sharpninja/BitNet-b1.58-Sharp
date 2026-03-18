using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.App;

public sealed record HostedAgentModelResponse(
    string Text,
    IReadOnlyList<string> Diagnostics);

public interface IHostedAgentModel : IDisposable
{
    string AgentName { get; }

    string ModelId { get; }

    string DisplayName { get; }

    string PrimaryLanguage { get; }

    VerbosityLevel Verbosity { get; }

    string SystemPrompt { get; }

    IReadOnlyList<string> DescribeModel();

    Task<HostedAgentModelResponse> GetResponseAsync(
        string prompt,
        int? maxOutputTokens = null,
        CancellationToken cancellationToken = default);
}

public interface IInspectableHostedAgentModel
{
    TernaryWeightStats GetTernaryWeightStats();
}

public interface ITrainableHostedAgentModel
{
    void Train(IEnumerable<TrainingExample> examples, int epochs = 1);
}
