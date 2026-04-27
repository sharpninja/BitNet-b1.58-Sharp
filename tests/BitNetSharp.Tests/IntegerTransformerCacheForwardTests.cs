using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F4 (float-deletion wiring): stack <see cref="IntegerForwardComposer.ForwardWithCache"/>
/// across every layer of a real <see cref="BitNetTransformer"/> so the
/// cache-aware decode hot loop has a fully integer alternative to
/// <see cref="BitNetTransformer.Forward(IReadOnlyList{int}, TransformerCache)"/>.
///
/// Cache seeding: both the float and int caches are prefilled through the
/// float path (<c>transformer.Forward(prefill, cache)</c>). Because the weights
/// and input are identical and prefill is deterministic, the two caches end up
/// bit-equal, giving a clean starting state to compare only the decode step.
/// </summary>
public sealed class IntegerTransformerCacheForwardTests
{
    [Fact]
    public void ForwardWithCacheInteger_SingleTokenDecode_MatchesFloatArgmax_TwoLayerMha()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 2,
            headCount: 2,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var transformer = new BitNetTransformer(
            config,
            NullLogger<BitNetTransformer>.Instance,
            seed: 233);

        var cacheFloat = transformer.CreateCache(capacity: 16);
        var cacheInt = transformer.CreateCache(capacity: 16);

        // Seed both caches identically via the float prefill path.
        var prefillTokens = new[] { 3, 11, 7, 19 };
        _ = transformer.Forward(prefillTokens, cacheFloat);
        _ = transformer.Forward(prefillTokens, cacheInt);

        var decodeTokens = new[] { 5 };
        float[,] floatLogits = transformer.Forward(decodeTokens, cacheFloat);
        float[,] intLogits = transformer.ForwardWithCacheInteger(decodeTokens, cacheInt);

        Assert.Equal(1, intLogits.GetLength(0));
        Assert.Equal(config.VocabSize, intLogits.GetLength(1));

        // Argmax on the decode row drives token selection. Softmax is
        // monotonic, so drift can move logits but must not cross the winner.
        int vocab = config.VocabSize;
        int floatArgmax = 0;
        int intArgmax = 0;
        float floatBest = floatLogits[0, 0];
        float intBest = intLogits[0, 0];
        for (int v = 1; v < vocab; v++)
        {
            if (floatLogits[0, v] > floatBest)
            {
                floatBest = floatLogits[0, v];
                floatArgmax = v;
            }
            if (intLogits[0, v] > intBest)
            {
                intBest = intLogits[0, v];
                intArgmax = v;
            }
        }
        Assert.Equal(floatArgmax, intArgmax);
    }

    [Fact]
    public void ForwardWithCacheInteger_SingleTokenDecode_PerElement_WithinTolerance_TwoLayerGqa()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var transformer = new BitNetTransformer(
            config,
            NullLogger<BitNetTransformer>.Instance,
            seed: 239);

        var cacheFloat = transformer.CreateCache(capacity: 16);
        var cacheInt = transformer.CreateCache(capacity: 16);

        var prefillTokens = new[] { 5, 13, 17, 23 };
        _ = transformer.Forward(prefillTokens, cacheFloat);
        _ = transformer.Forward(prefillTokens, cacheInt);

        var decodeTokens = new[] { 9 };
        float[,] floatLogits = transformer.Forward(decodeTokens, cacheFloat);
        float[,] intLogits = transformer.ForwardWithCacheInteger(decodeTokens, cacheInt);

        for (var c = 0; c < config.VocabSize; c++)
        {
            // Two stacked layers plus FinalNorm plus OutputHead: compounded
            // integer-precision floor stays inside 5e-1.
            Assert.InRange(intLogits[0, c] - floatLogits[0, c], -5e-1f, 5e-1f);
        }
    }

    [Fact]
    public void ForwardWithCacheInteger_AdvancesPastLength()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var transformer = new BitNetTransformer(
            config,
            NullLogger<BitNetTransformer>.Instance,
            seed: 241);
        var cache = transformer.CreateCache(capacity: 16);

        _ = transformer.ForwardWithCacheInteger(new[] { 3, 11, 7 }, cache);
        Assert.Equal(3, cache.PastLength);

        _ = transformer.ForwardWithCacheInteger(new[] { 19 }, cache);
        Assert.Equal(4, cache.PastLength);
    }
}
