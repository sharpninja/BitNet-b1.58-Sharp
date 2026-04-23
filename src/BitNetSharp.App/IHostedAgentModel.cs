using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;
using Microsoft.Extensions.AI;

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

    /// <summary>
    /// Streams assistant-side tokens for a multi-turn conversation. Default implementation
    /// flattens history to a single prompt via <see cref="PromptTemplate.FlattenHistory"/>,
    /// delegates to <see cref="GetResponseAsync"/>, and emits the final text as one chunk.
    /// Implementations with native streaming should override.
    /// </summary>
    async IAsyncEnumerable<string> StreamResponseAsync(
        IReadOnlyList<ChatMessage> history,
        int? maxOutputTokens = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(history);
        var prompt = PromptTemplate.FlattenHistory(SystemPrompt, history);
        var response = await GetResponseAsync(prompt, maxOutputTokens, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        if (!string.IsNullOrEmpty(response.Text))
        {
            yield return response.Text;
        }
    }
}

public interface IInspectableHostedAgentModel
{
    TernaryWeightStats GetTernaryWeightStats();
}

public interface ITrainableHostedAgentModel
{
    TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 1);
}
