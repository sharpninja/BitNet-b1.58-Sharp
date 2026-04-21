using System.Runtime.CompilerServices;
using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;
using Microsoft.Extensions.AI;

namespace BitNetSharp.Tests;

public sealed class HostedModelChatClientTests
{
    [Fact]
    public async Task chat_client_preserves_multi_turn_history()
    {
        var model = new RecordingModel(
            streamChunks: _ => new[] { "ok" });

        var client = new HostedModelChatClient(model);
        var history = new[]
        {
            new ChatMessage(ChatRole.User, "turn-one"),
            new ChatMessage(ChatRole.Assistant, "answer-one"),
            new ChatMessage(ChatRole.User, "turn-two")
        };

        await client.GetResponseAsync(history);

        var captured = Assert.Single(model.ReceivedPrompts);
        Assert.Contains("turn-one", captured);
        Assert.Contains("answer-one", captured);
        Assert.Contains("turn-two", captured);
    }

    [Fact]
    public async Task chat_client_streams_from_model_not_whitespace_split()
    {
        // Chunks deliberately contain NO whitespace splits that would match the old impl.
        var modelChunks = new[] { "hello-world-", "in-one-token" };
        var model = new RecordingModel(streamChunks: _ => modelChunks);
        var client = new HostedModelChatClient(model);

        var received = new List<string>();
        await foreach (var update in client.GetStreamingResponseAsync(new[] { new ChatMessage(ChatRole.User, "hi") }))
        {
            received.Add(update.Text ?? string.Empty);
        }

        Assert.Equal(modelChunks, received);
    }

    [Fact]
    public async Task chat_client_passes_max_output_tokens()
    {
        var model = new RecordingModel(streamChunks: _ => new[] { "ok" });
        var client = new HostedModelChatClient(model);
        var options = new ChatOptions { MaxOutputTokens = 42 };

        await foreach (var _ in client.GetStreamingResponseAsync(
            new[] { new ChatMessage(ChatRole.User, "hi") },
            options))
        {
        }

        Assert.Equal(42, model.LastMaxOutputTokens);
    }

    [Fact]
    public async Task chat_client_honors_cancellation_mid_stream()
    {
        using var cts = new CancellationTokenSource();
        var model = new RecordingModel(streamChunks: _ => InfiniteStream(cts));
        var client = new HostedModelChatClient(model);

        var received = new List<string>();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var update in client.GetStreamingResponseAsync(
                new[] { new ChatMessage(ChatRole.User, "hi") },
                options: null,
                cancellationToken: cts.Token))
            {
                received.Add(update.Text ?? string.Empty);
                if (received.Count == 2)
                {
                    cts.Cancel();
                }
            }
        });

        Assert.True(received.Count <= 3, $"expected cancellation to cap emission, got {received.Count}");
    }

    private static IEnumerable<string> InfiniteStream(CancellationTokenSource cts)
    {
        var i = 0;
        while (i < 1000)
        {
            yield return $"chunk-{i++}";
        }
    }

    private sealed class RecordingModel : IHostedAgentModel
    {
        private readonly Func<IReadOnlyList<ChatMessage>, IEnumerable<string>> _streamFactory;

        public RecordingModel(Func<IReadOnlyList<ChatMessage>, IEnumerable<string>> streamChunks)
        {
            _streamFactory = streamChunks;
        }

        public List<string> ReceivedPrompts { get; } = new();
        public int? LastMaxOutputTokens { get; private set; }

        public string AgentName => "recording";
        public string ModelId => "recording-model";
        public string DisplayName => "Recording stub";
        public string PrimaryLanguage => "en-US";
        public VerbosityLevel Verbosity => VerbosityLevel.Normal;
        public string SystemPrompt => "sys";

        public IReadOnlyList<string> DescribeModel() => Array.Empty<string>();

        public Task<HostedAgentModelResponse> GetResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            CancellationToken cancellationToken = default)
        {
            ReceivedPrompts.Add(prompt);
            LastMaxOutputTokens = maxOutputTokens;
            return Task.FromResult(new HostedAgentModelResponse(
                string.Concat(_streamFactory(Array.Empty<ChatMessage>())),
                Array.Empty<string>()));
        }

        public async IAsyncEnumerable<string> StreamResponseAsync(
            IReadOnlyList<ChatMessage> history,
            int? maxOutputTokens = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            LastMaxOutputTokens = maxOutputTokens;
            ReceivedPrompts.Add(PromptTemplate.FlattenHistory(SystemPrompt, history));
            foreach (var chunk in _streamFactory(history))
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return chunk;
                await Task.Yield();
            }
        }

        public void Dispose() { }
    }
}
