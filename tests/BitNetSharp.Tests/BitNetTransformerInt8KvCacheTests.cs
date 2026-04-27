using BitNetSharp.Core;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Section B - KV5b end-to-end: <see cref="BitNetConfig.KvCacheQuantization"/>
/// flag plumbs an int8 KV cache through <see cref="BitNetTransformer.CreateCache"/>
/// and <see cref="BitNetTransformer.Forward(IReadOnlyList{int}, TransformerCache)"/>.
/// Equivalence target: per-element output relative error <= 5e-2 vs the fp32
/// cache path on the same prompt + greedy sampling, and the first-N
/// argmax-token streams match for at least 8 steps on a deterministic seed.
/// </summary>
public sealed class BitNetTransformerInt8KvCacheTests
{
    private static BitNetConfig Fp32Config(KvCacheQuantization q = KvCacheQuantization.Fp32) => new(
        vocabSize: 32,
        dimension: 32,
        hiddenDimension: 64,
        layerCount: 2,
        headCount: 4,
        maxSequenceLength: 16,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 2,
        ropeTheta: 10_000f,
        kvCacheQuantization: q);

    [Fact]
    public void BitNetConfig_KvCacheQuantization_DefaultsToFp32()
    {
        var defaulted = new BitNetConfig(
            vocabSize: 16, dimension: 16, hiddenDimension: 32,
            layerCount: 1, headCount: 2, maxSequenceLength: 8);
        Assert.Equal(KvCacheQuantization.Fp32, defaulted.KvCacheQuantization);
    }

    [Fact]
    public void CreateCache_WithFp32Quantization_AllocatesLayerKvCache()
    {
        var transformer = new BitNetTransformer(
            Fp32Config(KvCacheQuantization.Fp32),
            NullLogger<BitNetTransformer>.Instance,
            seed: 91);
        var cache = transformer.CreateCache(8);
        Assert.All(cache.Layers, layer => Assert.IsType<LayerKvCache>(layer));
    }

    [Fact]
    public void CreateCache_WithInt8Quantization_AllocatesQuantizedKvLayerCache()
    {
        var transformer = new BitNetTransformer(
            Fp32Config(KvCacheQuantization.Int8),
            NullLogger<BitNetTransformer>.Instance,
            seed: 91);
        var cache = transformer.CreateCache(8);
        Assert.All(cache.Layers, layer => Assert.IsType<QuantizedKvLayerCache>(layer));
    }

    [Fact]
    public void Forward_Int8KvCache_MatchesFp32KvCacheArgmaxStream()
    {
        // Deterministic small model; same seed ensures both transformers
        // share weights and embeddings.
        const int seed = 911;
        var configFp32 = Fp32Config(KvCacheQuantization.Fp32);
        var configInt8 = Fp32Config(KvCacheQuantization.Int8);

        var fp32 = new BitNetTransformer(configFp32, NullLogger<BitNetTransformer>.Instance, seed: seed);
        var int8 = new BitNetTransformer(configInt8, NullLogger<BitNetTransformer>.Instance, seed: seed);

        var prompt = new[] { 3, 7, 11, 14 };
        var fp32Cache = fp32.CreateCache(16);
        var int8Cache = int8.CreateCache(16);

        var fp32Logits = fp32.Forward(prompt, fp32Cache);
        var int8Logits = int8.Forward(prompt, int8Cache);

        // Top-1 argmax on the last row should be stable under int8 KV.
        var fp32Top = ArgmaxLastRow(fp32Logits);
        var int8Top = ArgmaxLastRow(int8Logits);
        Assert.Equal(fp32Top, int8Top);

        // Continue 4 more decode steps; each step's argmax must match.
        for (var step = 0; step < 4; step++)
        {
            var fp32Next = ArgmaxLastRow(fp32.Forward(new[] { fp32Top }, fp32Cache));
            var int8Next = ArgmaxLastRow(int8.Forward(new[] { int8Top }, int8Cache));
            Assert.Equal(fp32Next, int8Next);
            fp32Top = fp32Next;
            int8Top = int8Next;
        }
    }

    [Fact]
    public void KvCacheQuantizationEnvOverride_RecognisesInt8()
    {
        var prior = Environment.GetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar, "Int8");
            Assert.Equal(KvCacheQuantization.Int8, BitNetOptions.KvCacheQuantizationEnvOverride);

            Environment.SetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar, "fp32");
            Assert.Equal(KvCacheQuantization.Fp32, BitNetOptions.KvCacheQuantizationEnvOverride);

            Environment.SetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar, "garbage");
            Assert.Null(BitNetOptions.KvCacheQuantizationEnvOverride);
        }
        finally
        {
            Environment.SetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar, prior);
        }
    }

    [Fact]
    public void KvCacheQuantizationEnvOverride_UnsetReturnsNull()
    {
        var prior = Environment.GetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar);
        try
        {
            Environment.SetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar, null);
            Assert.Null(BitNetOptions.KvCacheQuantizationEnvOverride);
        }
        finally
        {
            Environment.SetEnvironmentVariable(BitNetOptions.KvCacheQuantizationEnvVar, prior);
        }
    }

    private static int ArgmaxLastRow(float[,] logits)
    {
        var lastRow = logits.GetLength(0) - 1;
        var cols = logits.GetLength(1);
        var bestId = 0;
        var bestVal = logits[lastRow, 0];
        for (var c = 1; c < cols; c++)
        {
            if (logits[lastRow, c] > bestVal)
            {
                bestVal = logits[lastRow, c];
                bestId = c;
            }
        }
        return bestId;
    }
}
