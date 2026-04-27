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
        await foreach (var chunk in StreamResponseAsync(prompt, maxOutputTokens, cancellationToken).ConfigureAwait(false))
        {
            yield return chunk;
        }
    }

    /// <summary>
    /// Streams assistant-side tokens for an already-assembled prompt. Default implementation
    /// blocks on <see cref="GetResponseAsync"/> and emits the full response in one chunk.
    /// Implementations with native per-token streaming should override this overload.
    /// </summary>
    async IAsyncEnumerable<string> StreamResponseAsync(
        string prompt,
        int? maxOutputTokens = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var response = await GetResponseAsync(prompt, maxOutputTokens, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        if (!string.IsNullOrEmpty(response.Text))
        {
            yield return response.Text;
        }
    }

    /// <summary>
    /// Section A2: streams per-token <see cref="GeneratedToken"/> events
    /// carrying TokenId, TokenText, Step, ForwardMs, SelectMs, DecodeMs.
    /// Default implementation degrades to <see cref="StreamResponseAsync(string, int?, CancellationToken)"/>
    /// and yields a single GeneratedToken per text chunk with zero timing
    /// (callers without native per-token timing get the text but lose the
    /// telemetry). Implementations backed by an in-process model should
    /// override and surface real per-token timing.
    /// </summary>
    async IAsyncEnumerable<GeneratedToken> StreamTokensAsync(
        string prompt,
        int? maxOutputTokens = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var step = 0;
        await foreach (var piece in StreamResponseAsync(prompt, maxOutputTokens, cancellationToken).ConfigureAwait(false))
        {
            yield return new GeneratedToken(
                TokenId: -1,
                TokenText: piece,
                Step: step++,
                ForwardMs: 0d,
                SelectMs: 0d,
                DecodeMs: 0d);
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
