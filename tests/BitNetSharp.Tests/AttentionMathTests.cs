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

    // Section B-KV3 - DotInt8 / AccumulateWeightedInt8.
    // Equivalence target: relative error <= 2/127 (one absmax quantisation
    // step per dimension), which is the inherent worst case for per-row
    // int8 absmax + per-element rounding.

    private static (sbyte[] qInt8, float scale) QuantiseAbsmax(ReadOnlySpan<float> src)
    {
        var maxAbs = 0f;
        for (var i = 0; i < src.Length; i++)
        {
            var a = MathF.Abs(src[i]);
            if (a > maxAbs)
            {
                maxAbs = a;
            }
        }
        var scale = maxAbs <= 0f ? 1f : maxAbs / 127f;
        var q = new sbyte[src.Length];
        for (var i = 0; i < src.Length; i++)
        {
            var v = (int)MathF.Round(src[i] / scale, MidpointRounding.AwayFromZero);
            q[i] = (sbyte)Math.Clamp(v, -127, 127);
        }
        return (q, scale);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(128)]
    public void DotInt8_OnArmHost_MatchesPortableFallback(int headDim)
    {
        if (!System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported)
        {
            // Skipped on non-ARM hosts (x86/x64 dev machines). The ARM
            // kernel uses AdvSimd.FusedMultiplyAdd which throws on x86;
            // equivalence on ARM is the gate that matters.
            return;
        }

        var qFloat = RandomVector(headDim, seed: 1733 + headDim);
        var kFloat = RandomVector(headDim, seed: 1741 + headDim);
        var (kInt8, kScale) = QuantiseAbsmax(kFloat);

        // The portable Vector.Widen path is the cross-platform reference.
        // ARM dispatch routes to DotInt8Arm; equivalence within fp rounding.
        var armResult = AttentionMath.DotInt8(qFloat, kInt8, kScale, headDim);

        // Compute the same dot via the scalar oracle to bound drift.
        var scalarRef = 0f;
        for (var i = 0; i < headDim; i++)
        {
            scalarRef += qFloat[i] * (kInt8[i] * kScale);
        }
        Assert.True(MathF.Abs(armResult - scalarRef) < 1e-3f,
            $"ARM DotInt8 drift {armResult - scalarRef}");
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(127)]
    [InlineData(128)]
    public void DotInt8_EqualsFp32DotWithinQuantizationError(int headDim)
    {
        var qFloat = RandomVector(headDim, seed: 911 + headDim);
        var kFloat = RandomVector(headDim, seed: 977 + headDim);
        var (kInt8, kScale) = QuantiseAbsmax(kFloat);

        var fp32 = ScalarDot(qFloat, kFloat, headDim);
        var int8Dot = AttentionMath.DotInt8(qFloat, kInt8, kScale, headDim);

        // Absolute error bounded by sum_i |q[i]| * (kScale / 2). Use a
        // generous 2x bound to cover RNG worst case + AwayFromZero rounding.
        var absQ = 0f;
        for (var i = 0; i < headDim; i++)
        {
            absQ += MathF.Abs(qFloat[i]);
        }
        var bound = absQ * kScale + 1e-4f;
        Assert.InRange(int8Dot - fp32, -bound, bound);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    [InlineData(127)]
    [InlineData(128)]
    public void AccumulateWeightedInt8_EqualsFp32WithinQuantizationError(int headDim)
    {
        var src = RandomVector(headDim, seed: 1009 + headDim);
        var (srcInt8, srcScale) = QuantiseAbsmax(src);
        var weight = 0.37f;

        var targetFp32 = new float[headDim];
        AttentionMath.AccumulateWeighted(targetFp32, src, weight, headDim);

        var targetInt8 = new float[headDim];
        AttentionMath.AccumulateWeightedInt8(targetInt8, srcInt8, srcScale, weight, headDim);

        var bound = MathF.Abs(weight) * srcScale + 1e-4f;
        for (var i = 0; i < headDim; i++)
        {
            Assert.InRange(targetInt8[i] - targetFp32[i], -bound, bound);
        }
    }
}
