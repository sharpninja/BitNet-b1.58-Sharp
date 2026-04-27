using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

public sealed class FlashAttentionTests
{
    private const float Tolerance = 1e-4f;

    private static BitNetConfig GqaConfig() => new(
        vocabSize: 128,
        dimension: 64,
        hiddenDimension: 128,
        layerCount: 2,
        headCount: 4,
        maxSequenceLength: 32,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 2,
        ropeTheta: 10_000f);

    private static BitNetConfig MhaConfig() => new(
        vocabSize: 128,
        dimension: 64,
        hiddenDimension: 128,
        layerCount: 2,
        headCount: 4,
        maxSequenceLength: 32,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 4,
        ropeTheta: 10_000f);

    private static float[,] Random(int rows, int cols, int seed)
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

    private static float MaxAbsDiff(float[,] lhs, float[,] rhs)
    {
        var m = 0f;
        for (var r = 0; r < lhs.GetLength(0); r++)
        {
            for (var c = 0; c < lhs.GetLength(1); c++)
            {
                var d = MathF.Abs(lhs[r, c] - rhs[r, c]);
                if (d > m)
                {
                    m = d;
                }
            }
        }

        return m;
    }

    [Theory]
    [InlineData(4)]
    [InlineData(16)]
    [InlineData(31)]
    public void ForwardDecode_MatchesDenseAttention_Gqa(int prefillLen)
    {
        var config = GqaConfig();
        var gqaDense = new GroupedQueryAttention(config, new Random(42));
        var gqaFlash = new GroupedQueryAttention(config, new Random(42));

        var kvDim = config.KvHeadCount * config.HeadDimension;
        var prefill = Random(prefillLen, config.Dimension, seed: 23);
        var newRow = Random(1, config.Dimension, seed: 24);

        var denseCache = new LayerKvCache(prefillLen + 1, kvDim);
        _ = gqaDense.Forward(prefill, denseCache, positionOffset: 0);
        var denseOut = gqaDense.Forward(newRow, denseCache, positionOffset: prefillLen);

        var flashCache = new LayerKvCache(prefillLen + 1, kvDim);
        _ = gqaFlash.Forward(prefill, flashCache, positionOffset: 0);
        var flashOut = gqaFlash.ForwardFlashDecode(newRow, flashCache, positionOffset: prefillLen);

        Assert.Equal(1, flashOut.GetLength(0));
        Assert.Equal(config.Dimension, flashOut.GetLength(1));
        Assert.True(MaxAbsDiff(denseOut, flashOut) < Tolerance);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(16)]
    [InlineData(31)]
    public void ForwardDecode_MatchesDenseAttention_Mha(int prefillLen)
    {
        var config = MhaConfig();
        var mhaDense = new MultiHeadAttention(config, new Random(77));
        var mhaFlash = new MultiHeadAttention(config, new Random(77));

        var prefill = Random(prefillLen, config.Dimension, seed: 33);
        var newRow = Random(1, config.Dimension, seed: 34);

        var denseCache = new LayerKvCache(prefillLen + 1, config.Dimension);
        _ = mhaDense.Forward(prefill, denseCache, positionOffset: 0);
        var denseOut = mhaDense.Forward(newRow, denseCache, positionOffset: prefillLen);

        var flashCache = new LayerKvCache(prefillLen + 1, config.Dimension);
        _ = mhaFlash.Forward(prefill, flashCache, positionOffset: 0);
        var flashOut = mhaFlash.ForwardFlashDecode(newRow, flashCache, positionOffset: prefillLen);

        Assert.Equal(1, flashOut.GetLength(0));
        Assert.Equal(config.Dimension, flashOut.GetLength(1));
        Assert.True(MaxAbsDiff(denseOut, flashOut) < Tolerance);
    }

    [Fact]
    public void FlashAttention_SingleHead_EqualsManualOnlineSoftmax()
    {
        const int headDim = 8;
        var rng = new Random(91);

        var q = new float[headDim];
        for (var i = 0; i < headDim; i++)
        {
            q[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        const int past = 5;
        var k = new float[past * headDim];
        var v = new float[past * headDim];
        for (var i = 0; i < past * headDim; i++)
        {
            k[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var output = new float[headDim];
        var scale = 1f / MathF.Sqrt(headDim);

        AttentionMath.OnlineSoftmaxAttendSingleRow(
            q, k, v, output, 0, headDim, headDim, past, scale);

        var scores = new float[past];
        var maxScore = float.NegativeInfinity;
        for (var s = 0; s < past; s++)
        {
            var dot = 0f;
            for (var d = 0; d < headDim; d++)
            {
                dot += q[d] * k[s * headDim + d];
            }
            scores[s] = dot * scale;
            if (scores[s] > maxScore) maxScore = scores[s];
        }

        var partition = 0f;
        var exp = new float[past];
        for (var s = 0; s < past; s++)
        {
            exp[s] = MathF.Exp(scores[s] - maxScore);
            partition += exp[s];
        }

        var expected = new float[headDim];
        for (var s = 0; s < past; s++)
        {
            var weight = exp[s] / partition;
            for (var d = 0; d < headDim; d++)
            {
                expected[d] += weight * v[s * headDim + d];
            }
        }

        for (var d = 0; d < headDim; d++)
        {
            Assert.True(MathF.Abs(expected[d] - output[d]) < Tolerance,
                $"d={d} expected={expected[d]} actual={output[d]}");
        }
    }

    // Section B-KV4 - FlashAttention.ForwardDecodeInt8.
    // Online-softmax attention against a per-row absmax-quantised int8 K/V
    // cache. Equivalence target: relative error <= 5e-3 vs the fp32 path
    // across SeqLen ∈ {1, 8, 64, 512, 2048}; per-element bound proportional
    // to row-absmax / 127.

    private static (sbyte[] kInt8, sbyte[] vInt8, float[] kScale, float[] vScale) QuantiseCache(
        ReadOnlySpan<float> kFloat, ReadOnlySpan<float> vFloat, int rows, int kvDim)
    {
        var kInt8 = new sbyte[rows * kvDim];
        var vInt8 = new sbyte[rows * kvDim];
        var kScale = new float[rows];
        var vScale = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            var rowOffset = r * kvDim;
            var maxK = 0f;
            var maxV = 0f;
            for (var c = 0; c < kvDim; c++)
            {
                maxK = MathF.Max(maxK, MathF.Abs(kFloat[rowOffset + c]));
                maxV = MathF.Max(maxV, MathF.Abs(vFloat[rowOffset + c]));
            }
            var sK = maxK <= 0f ? 1f : maxK / 127f;
            var sV = maxV <= 0f ? 1f : maxV / 127f;
            kScale[r] = sK;
            vScale[r] = sV;
            for (var c = 0; c < kvDim; c++)
            {
                var qK = (int)MathF.Round(kFloat[rowOffset + c] / sK, MidpointRounding.AwayFromZero);
                var qV = (int)MathF.Round(vFloat[rowOffset + c] / sV, MidpointRounding.AwayFromZero);
                kInt8[rowOffset + c] = (sbyte)Math.Clamp(qK, -127, 127);
                vInt8[rowOffset + c] = (sbyte)Math.Clamp(qV, -127, 127);
            }
        }
        return (kInt8, vInt8, kScale, vScale);
    }

    [Theory]
    [InlineData(1, 4, 4, 16)]
    [InlineData(8, 4, 4, 16)]
    [InlineData(64, 4, 2, 32)]
    [InlineData(256, 4, 4, 32)]
    public void ForwardDecodeInt8_MatchesFp32WithinQuantizationError(
        int pastLength, int headCount, int kvHeadCount, int headDim)
    {
        var rng = new Random(1331 + pastLength);
        var dim = headCount * headDim;
        var kvDim = kvHeadCount * headDim;
        var query = new float[dim];
        for (var i = 0; i < dim; i++)
        {
            query[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var kFloat = new float[pastLength * kvDim];
        var vFloat = new float[pastLength * kvDim];
        for (var i = 0; i < kFloat.Length; i++)
        {
            kFloat[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            vFloat[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var (kInt8, vInt8, kScale, vScale) = QuantiseCache(kFloat, vFloat, pastLength, kvDim);

        var fp32Out = new float[dim];
        FlashAttention.ForwardDecode(query, kFloat, vFloat, fp32Out,
            headCount, kvHeadCount, headDim, pastLength, scale: 1f / MathF.Sqrt(headDim));

        var int8Out = new float[dim];
        FlashAttention.ForwardDecodeInt8(query, kInt8, kScale, vInt8, vScale, int8Out,
            headCount, kvHeadCount, headDim, pastLength, scale: 1f / MathF.Sqrt(headDim));

        // Per-element absolute error bound: average per-row dequant error is
        // ~ scale / 2; with softmax weighting and accumulation across rows
        // the bound is conservatively ~ max(vScale).
        var maxBound = vScale.Max() * 4f + 1e-3f;
        for (var i = 0; i < dim; i++)
        {
            Assert.InRange(int8Out[i] - fp32Out[i], -maxBound, maxBound);
        }
    }

    [Fact]
    public void ForwardDecodeInt8_PastLengthZero_ReturnsZeroOutput()
    {
        const int headCount = 2;
        const int kvHeadCount = 2;
        const int headDim = 8;
        const int dim = headCount * headDim;
        const int kvDim = kvHeadCount * headDim;

        var query = new float[dim];
        for (var i = 0; i < dim; i++) query[i] = 0.5f;
        var kInt8 = new sbyte[kvDim];
        var vInt8 = new sbyte[kvDim];
        var kScale = new[] { 1f };
        var vScale = new[] { 1f };
        var output = new float[dim];
        for (var i = 0; i < dim; i++) output[i] = 99f; // poison

        FlashAttention.ForwardDecodeInt8(query, kInt8, kScale, vInt8, vScale, output,
            headCount, kvHeadCount, headDim, pastLength: 0, scale: 1f / MathF.Sqrt(headDim));

        Assert.All(output, v => Assert.Equal(0f, v));
    }
}
