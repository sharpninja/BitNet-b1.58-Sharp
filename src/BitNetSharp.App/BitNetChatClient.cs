using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace BitNetSharp.App;

/// <summary>
/// <see cref="IChatClient"/> over <see cref="IHostedAgentModel"/>. Preserves multi-turn
/// history (does NOT drop to last-user-message) and delegates streaming to the model's
/// <see cref="IHostedAgentModel.StreamResponseAsync"/> rather than whitespace-splitting
/// a pre-materialized response.
/// </summary>
public sealed class HostedModelChatClient(IHostedAgentModel model) : IChatClient
{
    private readonly IHostedAgentModel _model = model ?? throw new ArgumentNullException(nameof(model));

    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var history = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();
        var prompt = PromptTemplate.FlattenHistory(_model.SystemPrompt, history);
        var result = await _model.GetResponseAsync(prompt, options?.MaxOutputTokens, cancellationToken)
            .ConfigureAwait(false);

        return new ChatResponse(new ChatMessage(ChatRole.Assistant, result.Text))
        {
            ModelId = _model.ModelId,
            FinishReason = ChatFinishReason.Stop,
            CreatedAt = DateTimeOffset.UtcNow
        };
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var history = messages as IReadOnlyList<ChatMessage> ?? messages.ToList();
        var createdAt = DateTimeOffset.UtcNow;

        await foreach (var chunk in _model
            .StreamResponseAsync(history, options?.MaxOutputTokens, cancellationToken)
            .ConfigureAwait(false))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrEmpty(chunk))
            {
                continue;
            }

            yield return new ChatResponseUpdate(ChatRole.Assistant, chunk)
            {
                ModelId = _model.ModelId,
                CreatedAt = createdAt
            };
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
