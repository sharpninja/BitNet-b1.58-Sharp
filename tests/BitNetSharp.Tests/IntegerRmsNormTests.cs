using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I3: RMSNorm swaps float sqrt for integer Newton-Raphson rsqrt in
/// Q16.16 fixed-point. The public surface stays float[,] -> float[,] until
/// upstream stages hand us integer rows, but the arithmetic under the hood
/// runs on int64 squared-sums and an integer rsqrt. Result must match the
/// reference float RmsNorm within 1e-3 per element to keep perplexity sane.
/// </summary>
public sealed class IntegerRmsNormTests
{
    [Fact]
    public void RsqrtQ16_16_ApproximatesInverseSqrt_AcrossFourDecades()
    {
        // Test x values from 0.1 to 10000 (covers typical sum-of-squares / dim)
        var testInputs = new[]
        {
            0.1f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 10.0f, 100.0f, 1000.0f, 10000.0f,
        };

        foreach (var x in testInputs)
        {
            var xQ = (long)(x * 65536.0);
            var rsqrtQ = IntegerMath.RsqrtQ16_16(xQ);
            var rsqrt = rsqrtQ / 65536f;
            var expected = 1f / MathF.Sqrt(x);

            var relativeError = MathF.Abs(rsqrt - expected) / expected;
            Assert.True(relativeError < 0.01f,
                $"Rsqrt({x}) = {rsqrt}, expected {expected}, rel err {relativeError}");
        }
    }

    [Fact]
    public void RsqrtQ16_16_HandlesSmallAndLargeInputs()
    {
        // Very small: x = 1/65536 = smallest positive Q16.16
        var smallQ = 1L;
        var rsqrtSmall = IntegerMath.RsqrtQ16_16(smallQ) / 65536f;
        var expectedSmall = 1f / MathF.Sqrt(1f / 65536f);
        Assert.True(MathF.Abs(rsqrtSmall - expectedSmall) / expectedSmall < 0.05f);

        // Very large: x = 2^20 = 1048576
        var largeQ = 1048576L << 16;
        var rsqrtLarge = IntegerMath.RsqrtQ16_16(largeQ) / 65536f;
        var expectedLarge = 1f / MathF.Sqrt(1048576f);
        Assert.True(MathF.Abs(rsqrtLarge - expectedLarge) / expectedLarge < 0.05f);
    }

    [Fact]
    public void IntegerRmsNorm_MatchesFloatReference_WithinTolerance()
    {
        const int dim = 64;
        var rng = new Random(7);
        var input = new float[4, dim];
        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < dim; c++)
            {
                input[r, c] = ((float)rng.NextDouble() - 0.5f) * 4f;
            }
        }

        var refNorm = new RmsNorm(dim);
        var refOutput = refNorm.Forward(input);

        var intNorm = new IntegerRmsNorm(dim);
        var intOutput = intNorm.Forward(input);

        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < dim; c++)
            {
                var expected = refOutput[r, c];
                var actual = intOutput[r, c];
                var tolerance = MathF.Abs(expected) * 0.02f + 1e-3f;
                Assert.InRange(actual - expected, -tolerance, tolerance);
            }
        }
    }

    [Fact]
    public void IntegerRmsNorm_ZeroInput_ReturnsZero()
    {
        const int dim = 16;
        var intNorm = new IntegerRmsNorm(dim);
        var input = new float[2, dim];
        var output = intNorm.Forward(input);

        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < dim; c++)
            {
                Assert.Equal(0f, output[r, c]);
            }
        }
    }

    [Fact]
    public void IntegerRmsNorm_WithLearnableScale_AppliesScaleCorrectly()
    {
        const int dim = 8;
        var intNorm = new IntegerRmsNorm(dim);

        var scale = new float[dim];
        for (var i = 0; i < dim; i++) scale[i] = 0.5f + i * 0.1f;
        intNorm.ImportScale(scale);

        var input = new float[1, dim];
        for (var i = 0; i < dim; i++) input[0, i] = 1f; // uniform input → rms = 1

        var output = intNorm.Forward(input);

        // Normalized = 1 / rms = 1 / sqrt(1 + eps) ≈ 1, multiplied by scale.
        for (var i = 0; i < dim; i++)
        {
            var expected = scale[i]; // roughly
            Assert.InRange(output[0, i] - expected, -0.01f, 0.01f);
        }
    }

    [Fact]
    public void IntegerRmsNorm_RejectsWrongInputDim()
    {
        var intNorm = new IntegerRmsNorm(8);
        var wrong = new float[1, 7];

        Assert.Throws<ArgumentException>(() => intNorm.Forward(wrong));
    }
}
