using Microsoft.Extensions.AI;

namespace BitNetSharp.App;

public sealed class HostedModelChatClient(IHostedAgentModel model) : IChatClient
{
    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var prompt = messages.LastOrDefault(message => message.Role == ChatRole.User)?.Text ?? string.Empty;
        var result = await model.GetResponseAsync(prompt, options?.MaxOutputTokens, cancellationToken);
        var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, result.Text))
        {
            ModelId = model.ModelId,
            FinishReason = ChatFinishReason.Stop,
            CreatedAt = DateTimeOffset.UtcNow
        };

        return response;
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken);

        foreach (var token in response.Text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, token)
            {
                ModelId = response.ModelId,
                CreatedAt = response.CreatedAt
            };
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null) => null;

    public void Dispose()
    {
    }
}
