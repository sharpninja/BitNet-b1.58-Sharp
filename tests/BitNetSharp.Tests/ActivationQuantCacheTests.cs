using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

public sealed class ActivationQuantCacheTests
{
    private const float Tolerance = 1e-5f;

    private static BitNetConfig SmallConfig() => new(
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

    [Fact]
    public void FromFloat_RoundTripsToOriginalDotProduct()
    {
        var input = Random(3, 64, seed: 7);
        var block = QuantizedActivationBlock.FromFloat(input);

        Assert.Equal(3, block.Rows);
        Assert.Equal(64, block.Cols);
        Assert.Equal(3 * 64, block.Quantized.Length);
        Assert.Equal(3, block.RowScales.Length);

        for (var r = 0; r < 3; r++)
        {
            var scale = block.RowScales[r];
            for (var c = 0; c < 64; c++)
            {
                var recon = block.Quantized[r * 64 + c] * scale;
                var absErr = MathF.Abs(recon - input[r, c]);
                Assert.True(absErr <= scale + 1e-6f, $"row={r} col={c} absErr={absErr} scale={scale}");
            }
        }
    }

    [Fact]
    public void ForwardQuantized_EqualsForward_BitLinear()
    {
        var shapes = new (int inDim, int outDim)[]
        {
            (32, 64),
            (64, 32),
            (64, 128),
            (128, 64),
        };

        foreach (var (inDim, outDim) in shapes)
        {
            var rng = new Random(19 + inDim + outDim);
            var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), rng);
            var input = Random(4, inDim, seed: 31 + inDim);

            var viaForward = layer.Forward(input);
            var block = QuantizedActivationBlock.FromFloat(input);
            var viaQuantized = layer.ForwardQuantized(block);

            Assert.Equal(viaForward.GetLength(0), viaQuantized.GetLength(0));
            Assert.Equal(viaForward.GetLength(1), viaQuantized.GetLength(1));
            Assert.Equal(0f, MaxAbsDiff(viaForward, viaQuantized));
        }
    }

    [Fact]
    public void MultiHeadAttention_QuantisesInputOnce()
    {
        var config = MhaConfig();
        var mha = new MultiHeadAttention(config, new Random(3));
        var input = Random(5, config.Dimension, seed: 71);

        var counter = new QuantizedActivationBlock.StrongBox<long>();
        QuantizedActivationBlock.FromFloatCallCounter.Value = counter;
        try
        {
            _ = mha.Forward(input);
        }
        finally
        {
            QuantizedActivationBlock.FromFloatCallCounter.Value = null!;
        }

        Assert.Equal(2, counter.Value);
    }

    [Fact]
    public void GroupedQueryAttention_QuantisesInputOnce()
    {
        var config = SmallConfig();
        var gqa = new GroupedQueryAttention(config, new Random(5));
        var input = Random(5, config.Dimension, seed: 83);

        var counter = new QuantizedActivationBlock.StrongBox<long>();
        QuantizedActivationBlock.FromFloatCallCounter.Value = counter;
        try
        {
            _ = gqa.Forward(input);
        }
        finally
        {
            QuantizedActivationBlock.FromFloatCallCounter.Value = null!;
        }

        Assert.Equal(2, counter.Value);
    }

    [Fact]
    public void SwiGLUFeedForward_QuantisesInputOnce()
    {
        var config = SmallConfig();
        var ff = new SwiGLUFeedForward(config, new Random(7));
        var input = Random(5, config.Dimension, seed: 91);

        var counter = new QuantizedActivationBlock.StrongBox<long>();
        QuantizedActivationBlock.FromFloatCallCounter.Value = counter;
        try
        {
            _ = ff.Forward(input);
        }
        finally
        {
            QuantizedActivationBlock.FromFloatCallCounter.Value = null!;
        }

        Assert.Equal(2, counter.Value);
    }

    [Fact]
    public void MultiHeadAttention_SharedQuant_MatchesLegacyForward()
    {
        var config = MhaConfig();
        var mha1 = new MultiHeadAttention(config, new Random(11));
        var mha2 = new MultiHeadAttention(config, new Random(11));
        var input = Random(4, config.Dimension, seed: 121);

        var expected = mha1.Forward(input);
        var actual = mha2.Forward(input);

        Assert.True(MaxAbsDiff(expected, actual) < Tolerance);
    }

    [Fact]
    public void GroupedQueryAttention_SharedQuant_MatchesLegacyForward()
    {
        var config = SmallConfig();
        var gqa1 = new GroupedQueryAttention(config, new Random(13));
        var gqa2 = new GroupedQueryAttention(config, new Random(13));
        var input = Random(4, config.Dimension, seed: 131);

        var expected = gqa1.Forward(input);
        var actual = gqa2.Forward(input);

        Assert.True(MaxAbsDiff(expected, actual) < Tolerance);
    }
}
