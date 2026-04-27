using System.Reflection;
using System.Text;
using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Section A2 of the residual close-out: <c>GeneratedToken</c> carries
/// per-token timing (ForwardMs / SelectMs / DecodeMs) and a new
/// <c>StreamTokensAsync</c> overload on <see cref="IHostedAgentModel"/>
/// surfaces the rich record. The legacy text-only
/// <c>StreamResponseAsync(string)</c> overload remains so existing serve
/// callers keep working unchanged.
/// </summary>
public sealed class GeneratedTokenStreamingTests
{
    private static readonly string[] Vocabulary =
    [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"
    ];

    [Fact]
    public void GeneratedToken_RecordHasTimingFields()
    {
        var properties = typeof(GeneratedToken).GetProperties(BindingFlags.Public | BindingFlags.Instance);
        var names = properties.Select(p => p.Name).ToHashSet();

        Assert.Contains(nameof(GeneratedToken.TokenId), names);
        Assert.Contains(nameof(GeneratedToken.TokenText), names);
        Assert.Contains(nameof(GeneratedToken.Step), names);
        Assert.Contains(nameof(GeneratedToken.ForwardMs), names);
        Assert.Contains(nameof(GeneratedToken.SelectMs), names);
        Assert.Contains(nameof(GeneratedToken.DecodeMs), names);
    }

    [Fact]
    public async Task StreamGenerateAsync_EmitsPerTokenTiming()
    {
        var model = BuildSmallModel();

        var emitted = new List<GeneratedToken>();
        await foreach (var token in model.StreamGenerateAsync("alpha beta gamma", maxTokens: 4))
        {
            emitted.Add(token);
        }

        Assert.NotEmpty(emitted);
        for (var i = 0; i < emitted.Count; i++)
        {
            var t = emitted[i];
            Assert.Equal(i, t.Step);
            Assert.True(t.ForwardMs > 0d, $"step {i} ForwardMs expected > 0 but was {t.ForwardMs:F3}");
            Assert.True(t.SelectMs >= 0d, $"step {i} SelectMs expected >= 0 but was {t.SelectMs:F3}");
            // Last token never triggered a follow-up decode (loop exited or
            // hit cap before the post-emit decode), so DecodeMs may be 0.
            Assert.True(t.DecodeMs >= 0d, $"step {i} DecodeMs expected >= 0 but was {t.DecodeMs:F3}");
            if (i < emitted.Count - 1)
            {
                Assert.True(t.DecodeMs > 0d, $"step {i} DecodeMs expected > 0 (mid-stream) but was {t.DecodeMs:F3}");
            }
        }
    }

    [Fact]
    public async Task StreamTokensAsync_HostedAgent_EmitsPerTokenTiming()
    {
        var model = BuildSmallModel();
        IHostedAgentModel agent = new BitNetHostedAgentModel(model);

        var emitted = new List<GeneratedToken>();
        await foreach (var token in agent.StreamTokensAsync("alpha beta gamma", maxOutputTokens: 4))
        {
            emitted.Add(token);
        }

        Assert.NotEmpty(emitted);
        Assert.All(emitted, t => Assert.True(t.ForwardMs > 0d, $"step {t.Step} ForwardMs expected > 0 but was {t.ForwardMs:F3}"));
    }

    [Fact]
    public async Task StreamResponseAsync_StringOverloadStillWorksUnchanged()
    {
        var model = BuildSmallModel();
        IHostedAgentModel agent = new BitNetHostedAgentModel(model);

        var sb = new StringBuilder();
        await foreach (var piece in agent.StreamResponseAsync("alpha beta gamma", maxOutputTokens: 4))
        {
            sb.Append(piece);
        }

        var streamed = sb.ToString();
        var direct = await agent.GetResponseAsync("alpha beta gamma", maxOutputTokens: 4);
        // Streaming and non-streaming produce the same final assistant text.
        Assert.Equal(direct.Text.TrimEnd(), streamed.TrimEnd());
    }

    private static BitNetPaperModel BuildSmallModel()
    {
        var options = new BitNetOptions(
            Vocabulary,
            VerbosityLevel.Quiet,
            MaxResponseTokens: 4,
            UseIntegerForward: false);
        var config = new BitNetConfig(
            vocabSize: 11,
            dimension: 32,
            hiddenDimension: 96,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        return new BitNetPaperModel(
            options,
            NullLogger<BitNetPaperModel>.Instance,
            NullLoggerFactory.Instance,
            config,
            seed: 271);
    }
}
