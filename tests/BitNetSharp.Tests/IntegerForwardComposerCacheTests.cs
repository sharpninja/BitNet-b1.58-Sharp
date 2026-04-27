using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F3 (float-deletion wiring, cache-aware decode): composes the integer
/// primitives into a cache-aware <see cref="BitNetLayer"/> forward pass so the
/// per-token decode hot loop can stop going through float attention. Covers
/// single-row decode (the hot path) and multi-row prefill-with-cache (the
/// warm path). Argmax-driven sampling is softmax-monotonic, so the tolerance
/// only needs to keep intermediate hidden states recoverable.
/// </summary>
public sealed class IntegerForwardComposerCacheTests
{
    [Fact]
    public void ForwardWithCache_SingleRowDecode_MatchesFloat_Mha()
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
        var rng = new Random(181);
        var layer = new BitNetLayer(config, rng);

        var cacheFloat = new LayerKvCache(capacity: 16, kvDimension: config.Dimension);
        var cacheInt = new LayerKvCache(capacity: 16, kvDimension: config.Dimension);

        // Seed both caches identically via the float prefill path. Both caches
        // receive the same input + weights + positionOffset=0, so K and V end
        // up bit-equal after this call.
        var prefillInput = BuildMatrix(3, config.Dimension, rng);
        _ = layer.Forward(prefillInput, cacheFloat, 0);
        _ = layer.Forward(prefillInput, cacheInt, 0);

        var decodeInput = BuildMatrix(1, config.Dimension, rng);
        float[,] floatOut = layer.Forward(decodeInput, cacheFloat, positionOffset: 3);
        float[,] intOut = IntegerForwardComposer.ForwardWithCache(layer, decodeInput, cacheInt, positionOffset: 3);

        Assert.Equal(1, intOut.GetLength(0));
        Assert.Equal(config.Dimension, intOut.GetLength(1));
        for (var c = 0; c < config.Dimension; c++)
        {
            Assert.InRange(intOut[0, c] - floatOut[0, c], -5e-2f, 5e-2f);
        }
    }

    [Fact]
    public void ForwardWithCache_SingleRowDecode_MatchesFloat_Gqa()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 1,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var rng = new Random(191);
        var layer = new BitNetLayer(config, rng);

        // GQA's K/V projection has kvDim = kvHeadCount * headDim = 2 * 16 = 32.
        int kvDim = config.KvHeadCount * config.HeadDimension;
        var cacheFloat = new LayerKvCache(capacity: 16, kvDimension: kvDim);
        var cacheInt = new LayerKvCache(capacity: 16, kvDimension: kvDim);

        var prefillInput = BuildMatrix(3, config.Dimension, rng);
        _ = layer.Forward(prefillInput, cacheFloat, 0);
        _ = layer.Forward(prefillInput, cacheInt, 0);

        var decodeInput = BuildMatrix(1, config.Dimension, rng);
        float[,] floatOut = layer.Forward(decodeInput, cacheFloat, positionOffset: 3);
        float[,] intOut = IntegerForwardComposer.ForwardWithCache(layer, decodeInput, cacheInt, positionOffset: 3);

        for (var c = 0; c < config.Dimension; c++)
        {
            Assert.InRange(intOut[0, c] - floatOut[0, c], -5e-2f, 5e-2f);
        }
    }

    [Fact]
    public void ForwardWithCache_MultiRowPrefill_MatchesFloat_Mha()
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
        var rng = new Random(193);
        var layer = new BitNetLayer(config, rng);

        var cacheFloat = new LayerKvCache(capacity: 16, kvDimension: config.Dimension);
        var cacheInt = new LayerKvCache(capacity: 16, kvDimension: config.Dimension);

        var prefillInput = BuildMatrix(4, config.Dimension, rng);
        float[,] floatOut = layer.Forward(prefillInput, cacheFloat, positionOffset: 0);
        float[,] intOut = IntegerForwardComposer.ForwardWithCache(layer, prefillInput, cacheInt, positionOffset: 0);

        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < config.Dimension; c++)
            {
                Assert.InRange(intOut[r, c] - floatOut[r, c], -5e-2f, 5e-2f);
            }
        }
    }

    private static float[,] BuildMatrix(int rows, int cols, Random rng)
    {
        var m = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                m[r, c] = ((float)rng.NextDouble() - 0.5f) * 2f;
            }
        }
        return m;
    }
}
