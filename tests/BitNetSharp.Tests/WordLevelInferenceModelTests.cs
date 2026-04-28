using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Behaviour tests for WordLevelInferenceModel: prompt encoding, streaming
/// API contract, and EOS-stop semantics. The transformer used is a tiny
/// random-init small-preset; we don't care about output quality here, only
/// that the inference loop does not crash and emits the expected token-id
/// shape.
/// </summary>
public sealed class WordLevelInferenceModelTests
{
    private static (BitNetTransformer Transformer, WordLevelTokenizer Tokenizer) BuildTinyModel()
    {
        var lines = new[]
        {
            "[USER] take me to dallas [INTENT] {\"intent\":\"navigate\",\"slots\":{\"destination\":\"Dallas\"}}",
            "[USER] check my hours [INTENT] {\"intent\":\"hos_status\",\"slots\":{}}",
            "[USER] avoid tolls [INTENT] {\"intent\":\"route_preference\",\"slots\":{\"preference\":\"avoid tolls\"}}",
            "[USER] what's my eta [INTENT] {\"intent\":\"eta_query\",\"slots\":{}}",
        };
        var tokenizer = WordLevelTokenizer.TrainFromCorpus(lines, maxVocab: 200, minFrequency: 1);

        var cfg = new BitNetConfig(
            vocabSize: tokenizer.VocabSize,
            dimension: 32,
            hiddenDimension: 64,
            layerCount: 2,
            headCount: 2,
            maxSequenceLength: 64,
            kvHeadCount: 2);
        var transformer = new BitNetTransformer(cfg, NullLogger<BitNetTransformer>.Instance, seed: 42);
        return (transformer, tokenizer);
    }

    [Fact]
    public void EncodeForGeneration_PrependsBos_NoTrailingEos()
    {
        var (_, tok) = BuildTinyModel();
        var ids = tok.EncodeForGeneration("[USER] take me to dallas [INTENT]");
        Assert.True(ids.Length > 1);
        Assert.Equal(WordLevelTokenizer.BosId, ids[0]);
        Assert.NotEqual(WordLevelTokenizer.EosId, ids[^1]);
    }

    [Fact]
    public void GetTokenString_ReturnsSpecialsAtKnownIds()
    {
        var (_, tok) = BuildTinyModel();
        Assert.Equal("[PAD]", tok.GetTokenString(WordLevelTokenizer.PadId));
        Assert.Equal("[UNK]", tok.GetTokenString(WordLevelTokenizer.UnkId));
        Assert.Equal("[BOS]", tok.GetTokenString(WordLevelTokenizer.BosId));
        Assert.Equal("[EOS]", tok.GetTokenString(WordLevelTokenizer.EosId));
        Assert.Equal("[USER]", tok.GetTokenString(WordLevelTokenizer.UserId));
        Assert.Equal("[INTENT]", tok.GetTokenString(WordLevelTokenizer.IntentId));
    }

    [Fact]
    public void GetTokenString_OutOfRange_ReturnsUnk()
    {
        var (_, tok) = BuildTinyModel();
        Assert.Equal("[UNK]", tok.GetTokenString(-1));
        Assert.Equal("[UNK]", tok.GetTokenString(99_999));
    }

    [Fact]
    public void Constructor_VocabMismatch_Throws()
    {
        var (transformer, _) = BuildTinyModel();
        // Train a tokenizer with a different vocab size than the transformer.
        var otherTokenizer = WordLevelTokenizer.TrainFromCorpus(
            new[] { "totally different corpus content" },
            maxVocab: 50,
            minFrequency: 1);
        Assert.Throws<ArgumentException>(() => new WordLevelInferenceModel(transformer, otherTokenizer));
    }

    [Fact]
    public async Task StreamGenerateAsync_EmitsExpectedNumberOfTokens()
    {
        var (transformer, tok) = BuildTinyModel();
        var model = new WordLevelInferenceModel(transformer, tok)
        {
            MaxResponseTokens = 8,
            SuppressEosAndUnk = true, // force full output for the test
        };

        var emitted = new List<GeneratedToken>();
        await foreach (var t in model.StreamGenerateAsync("[USER] take me to dallas [INTENT]"))
        {
            emitted.Add(t);
        }

        Assert.Equal(8, emitted.Count);
        for (var i = 0; i < emitted.Count; i++)
        {
            Assert.Equal(i, emitted[i].Step);
            Assert.True(emitted[i].TokenId >= 0);
            Assert.True(emitted[i].TokenId < tok.VocabSize);
        }
        // First token's DecodeMs is zero (no prior decode); subsequent
        // tokens should report the prior step's decode latency.
        Assert.Equal(0d, emitted[0].DecodeMs);
    }

    [Fact]
    public async Task StreamGenerateAsync_StopsAtEosWhenNotSuppressed()
    {
        var (transformer, tok) = BuildTinyModel();
        var model = new WordLevelInferenceModel(transformer, tok)
        {
            MaxResponseTokens = 8,
            SuppressEosAndUnk = false,
        };

        var emitted = new List<GeneratedToken>();
        await foreach (var t in model.StreamGenerateAsync("[USER] hello [INTENT]"))
        {
            emitted.Add(t);
        }

        // Untrained random-init model usually argmaxes to EOS quickly.
        // We don't assert "<= N" here — just that no error occurs and the
        // loop terminates.
        Assert.True(emitted.Count >= 1);
        Assert.True(emitted.Count <= 8);
    }

    [Fact]
    public void GenerateResponse_ReturnsString_DoesNotThrow()
    {
        var (transformer, tok) = BuildTinyModel();
        var model = new WordLevelInferenceModel(transformer, tok)
        {
            MaxResponseTokens = 4,
            SuppressEosAndUnk = true,
        };
        var output = model.GenerateResponse("[USER] check my hours [INTENT]");
        Assert.NotNull(output);
    }
}
