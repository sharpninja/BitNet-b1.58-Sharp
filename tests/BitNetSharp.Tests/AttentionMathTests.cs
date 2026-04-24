using BitNetSharp.Core.Layers;

namespace BitNetSharp.Tests;

public sealed class AttentionMathTests
{
    private const float Tolerance = 1e-4f;

    private static float[] RandomVector(int length, int seed)
    {
        var rng = new Random(seed);
        var buffer = new float[length];
        for (var i = 0; i < length; i++)
        {
            buffer[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        return buffer;
    }

    private static float ScalarDot(ReadOnlySpan<float> a, ReadOnlySpan<float> b, int n)
    {
        var s = 0f;
        for (var i = 0; i < n; i++)
        {
            s += a[i] * b[i];
        }

        return s;
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(127)]
    [InlineData(128)]
    public void Dot_EqualsScalarOracle(int headDim)
    {
        var a = RandomVector(headDim, seed: headDim + 1);
        var b = RandomVector(headDim, seed: headDim + 2);

        var expected = ScalarDot(a, b, headDim);
        var actual = AttentionMath.Dot(a, b, headDim);

        Assert.True(MathF.Abs(expected - actual) < Tolerance, $"headDim={headDim} expected={expected} actual={actual}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(127)]
    [InlineData(128)]
    public void AccumulateWeighted_EqualsScalarOracle(int headDim)
    {
        var source = RandomVector(headDim, seed: headDim + 3);
        var initial = RandomVector(headDim, seed: headDim + 4);
        const float weight = 0.37f;

        var expected = new float[headDim];
        var actual = new float[headDim];
        initial.CopyTo(expected, 0);
        initial.CopyTo(actual, 0);
        for (var i = 0; i < headDim; i++)
        {
            expected[i] += weight * source[i];
        }

        AttentionMath.AccumulateWeighted(actual, source, weight, headDim);

        for (var i = 0; i < headDim; i++)
        {
            Assert.True(MathF.Abs(expected[i] - actual[i]) < Tolerance,
                $"headDim={headDim} i={i} expected={expected[i]} actual={actual[i]}");
        }
    }

    [Fact]
    public void Dot_WiderBuffer_OnlySumsHeadSlice()
    {
        var headDim = 16;
        var q = RandomVector(64, seed: 19);
        var k = RandomVector(64, seed: 23);

        var expected = ScalarDot(q, k, headDim);
        var actual = AttentionMath.Dot(q, k, headDim);

        Assert.True(MathF.Abs(expected - actual) < Tolerance);
    }
}
