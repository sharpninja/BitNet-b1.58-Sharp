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
}
