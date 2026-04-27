using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

/// <summary>
/// Section B (quantized KV cache) - KV1 red tests for the int8 K/V buffer.
/// QuantizedKvLayerCache stores per-row absmax-quantised sbyte K/V plus a
/// per-row float scale. WriteRow must round-trip within the inherent
/// quantisation error, handle the all-zero-row edge case with a sentinel
/// scale matching the existing QuantizedActivationBlock contract, and not
/// overflow at the absmax boundary.
/// </summary>
public sealed class QuantizedKvCacheTests
{
    [Fact]
    public void WriteRow_RoundTripsWithinAbsmaxQuantizationError()
    {
        const int capacity = 4;
        const int kvDim = 64;
        var cache = new QuantizedKvLayerCache(capacity, kvDim);

        var rng = new Random(101);
        var kRow = new float[kvDim];
        var vRow = new float[kvDim];
        for (var i = 0; i < kvDim; i++)
        {
            kRow[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vRow[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
        }

        cache.WriteRow(2, kRow, vRow);

        var kBack = new float[kvDim];
        var vBack = new float[kvDim];
        cache.DequantizeKRow(2, kBack);
        cache.DequantizeVRow(2, vBack);

        // Absmax int8 quantization: per-element error <= scale where scale = max|x|/127.
        // We use scale (not scale/2) as the bound to allow for the rounding
        // direction (AwayFromZero) plus the boundary clamp.
        var kScaleBound = MathF.Abs(kRow.Max(MathF.Abs)) / 127f * 1.001f;
        var vScaleBound = MathF.Abs(vRow.Max(MathF.Abs)) / 127f * 1.001f;
        for (var i = 0; i < kvDim; i++)
        {
            Assert.InRange(kBack[i] - kRow[i], -kScaleBound, kScaleBound);
            Assert.InRange(vBack[i] - vRow[i], -vScaleBound, vScaleBound);
        }
    }

    [Fact]
    public void WriteRow_HandlesZeroRow()
    {
        var cache = new QuantizedKvLayerCache(capacity: 2, kvDimension: 16);
        var kZero = new float[16];
        var vZero = new float[16];

        cache.WriteRow(0, kZero, vZero);

        // Sentinel scale = 1f matches the existing QuantizedActivationBlock
        // all-zero-row contract; dequant of all-zero sbyte yields zero.
        Assert.Equal(1f, cache.KScale[0]);
        Assert.Equal(1f, cache.VScale[0]);

        var kBack = new float[16];
        var vBack = new float[16];
        cache.DequantizeKRow(0, kBack);
        cache.DequantizeVRow(0, vBack);
        Assert.All(kBack, v => Assert.Equal(0f, v));
        Assert.All(vBack, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void WriteRow_HandlesAbsmaxAtBoundary()
    {
        var cache = new QuantizedKvLayerCache(capacity: 1, kvDimension: 8);
        // max element = 127 * scale where scale = 1f exactly: scale = 127/127 = 1f.
        var kRow = new float[] { -127f, -64f, 0f, 1f, 32f, 63f, 100f, 127f };
        var vRow = new float[8];
        Array.Copy(kRow, vRow, 8);

        cache.WriteRow(0, kRow, vRow);

        // Boundary element must clamp to sbyte ±127, not overflow.
        Assert.Equal((sbyte)-127, cache.K[0, 0]);
        Assert.Equal((sbyte)127, cache.K[0, 7]);
        Assert.Equal(1f, cache.KScale[0]);
    }

    [Fact]
    public void Constructor_RejectsBadDimensions()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new QuantizedKvLayerCache(0, 64));
        Assert.Throws<ArgumentOutOfRangeException>(() => new QuantizedKvLayerCache(4, 0));
    }

    [Fact]
    public void IKvCache_Fp32AndInt8_BothImplementWriteContract()
    {
        const int capacity = 4;
        const int kvDim = 16;
        var fp32 = new LayerKvCache(capacity, kvDim);
        var int8 = new QuantizedKvLayerCache(capacity, kvDim);

        var rng = new Random(53);
        var kRow = new float[kvDim];
        var vRow = new float[kvDim];
        for (var i = 0; i < kvDim; i++)
        {
            kRow[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vRow[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        IKvCache fp32Iface = fp32;
        IKvCache int8Iface = int8;
        Assert.Equal(capacity, fp32Iface.Capacity);
        Assert.Equal(kvDim, fp32Iface.KvDimension);
        Assert.Equal(capacity, int8Iface.Capacity);
        Assert.Equal(kvDim, int8Iface.KvDimension);

        fp32Iface.WriteKRow(1, kRow);
        fp32Iface.WriteVRow(1, vRow);
        int8Iface.WriteKRow(1, kRow);
        int8Iface.WriteVRow(1, vRow);

        // Fp32 path stores exact values.
        for (var i = 0; i < kvDim; i++)
        {
            Assert.Equal(kRow[i], fp32.K[1, i]);
            Assert.Equal(vRow[i], fp32.V[1, i]);
        }

        // Int8 path round-trips within scale.
        var kBack = new float[kvDim];
        int8.DequantizeKRow(1, kBack);
        var bound = MathF.Abs(kRow.Max(MathF.Abs)) / 127f * 1.001f;
        for (var i = 0; i < kvDim; i++)
        {
            Assert.InRange(kBack[i] - kRow[i], -bound, bound);
        }
    }

    // KV5 - wire QuantizedKvLayerCache into MHA and GQA cache-aware paths.

    private static BitNetConfig SmallMhaConfig() => new(
        vocabSize: 64,
        dimension: 32,
        hiddenDimension: 64,
        layerCount: 1,
        headCount: 4,
        maxSequenceLength: 16,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 4,
        ropeTheta: 10_000f);

    private static BitNetConfig SmallGqaConfig() => new(
        vocabSize: 64,
        dimension: 32,
        hiddenDimension: 64,
        layerCount: 1,
        headCount: 4,
        maxSequenceLength: 16,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 2,
        ropeTheta: 10_000f);

    private static float[,] RandomActivations(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var buffer = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                buffer[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
        return buffer;
    }

    [Fact]
    public void Mha_QuantizedCache_FlashDecode_MatchesFp32CacheWithinTolerance()
    {
        var config = SmallMhaConfig();
        var mha = new MultiHeadAttention(config, new Random(53));
        var prefill = RandomActivations(8, config.Dimension, seed: 71);
        var newRow = RandomActivations(1, config.Dimension, seed: 73);

        var fp32Cache = new LayerKvCache(16, config.Dimension);
        _ = mha.Forward(prefill, fp32Cache, positionOffset: 0);
        var fp32Out = mha.ForwardFlashDecode(newRow, fp32Cache, positionOffset: 8);

        var int8Cache = new QuantizedKvLayerCache(16, config.Dimension);
        _ = mha.Forward(prefill, int8Cache, positionOffset: 0);
        var int8Out = mha.ForwardFlashDecode(newRow, int8Cache, positionOffset: 8);

        Assert.Equal(1, int8Out.GetLength(0));
        Assert.Equal(config.Dimension, int8Out.GetLength(1));
        for (var c = 0; c < config.Dimension; c++)
        {
            Assert.InRange(int8Out[0, c] - fp32Out[0, c], -0.05f, 0.05f);
        }
    }

    [Fact]
    public void Gqa_QuantizedCache_FlashDecode_MatchesFp32CacheWithinTolerance()
    {
        var config = SmallGqaConfig();
        var gqa = new GroupedQueryAttention(config, new Random(59));
        var kvDim = config.KvHeadCount * config.HeadDimension;
        var prefill = RandomActivations(8, config.Dimension, seed: 79);
        var newRow = RandomActivations(1, config.Dimension, seed: 83);

        var fp32Cache = new LayerKvCache(16, kvDim);
        _ = gqa.Forward(prefill, fp32Cache, positionOffset: 0);
        var fp32Out = gqa.ForwardFlashDecode(newRow, fp32Cache, positionOffset: 8);

        var int8Cache = new QuantizedKvLayerCache(16, kvDim);
        _ = gqa.Forward(prefill, int8Cache, positionOffset: 0);
        var int8Out = gqa.ForwardFlashDecode(newRow, int8Cache, positionOffset: 8);

        Assert.Equal(1, int8Out.GetLength(0));
        Assert.Equal(config.Dimension, int8Out.GetLength(1));
        for (var c = 0; c < config.Dimension; c++)
        {
            Assert.InRange(int8Out[0, c] - fp32Out[0, c], -0.05f, 0.05f);
        }
    }
}
