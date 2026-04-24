using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

/// <summary>
/// Correctness harness for the fused pack-native ternary dot product.
/// <see cref="TritPacking.TernaryDotPacked"/> must be numerically identical
/// to "unpack the row first, then scalar dot" for every input it is called
/// with. The BitLinear hot-path optimization swaps the unpack+sbyte-span
/// TernaryDot pair for this fused variant, so any drift here silently
/// corrupts 8B-param inference.
/// </summary>
public sealed class TritPackingTernaryDotTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(5, 3)]
    [InlineData(7, 4)]
    [InlineData(11, 5)]
    [InlineData(64, 6)]
    [InlineData(127, 7)]
    [InlineData(256, 8)]
    [InlineData(1024, 9)]
    [InlineData(4096, 10)]
    public void TernaryDotPacked_EqualsUnpackThenScalarDot(int totalTrits, int seed)
    {
        var random = new Random(seed);

        var trits = new sbyte[totalTrits];
        var activations = new sbyte[totalTrits];
        for (var i = 0; i < totalTrits; i++)
        {
            trits[i] = (sbyte)(random.Next(3) - 1);
            activations[i] = (sbyte)(random.Next(255) - 127);
        }

        var packed = TritPacking.PackLayer(trits);
        var packedStride = (totalTrits + 4) / 5;

        var expected = 0;
        for (var i = 0; i < totalTrits; i++)
        {
            var w = trits[i];
            if (w > 0)
            {
                expected += activations[i];
            }
            else if (w < 0)
            {
                expected -= activations[i];
            }
        }

        var actual = TritPacking.TernaryDotPacked(
            packed.AsSpan(0, packedStride),
            activations,
            totalTrits);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TernaryDotPacked_AllZeroTrits_ReturnsZero()
    {
        var trits = new sbyte[100];
        var activations = new sbyte[100];
        var random = new Random(123);
        for (var i = 0; i < activations.Length; i++)
        {
            activations[i] = (sbyte)(random.Next(255) - 127);
        }

        var packed = TritPacking.PackLayer(trits);
        var packedStride = (100 + 4) / 5;

        var actual = TritPacking.TernaryDotPacked(
            packed.AsSpan(0, packedStride),
            activations,
            100);

        Assert.Equal(0, actual);
    }

    [Fact]
    public void TernaryDotPacked_AllPositiveTrits_SumsActivations()
    {
        var trits = new sbyte[50];
        var activations = new sbyte[50];
        for (var i = 0; i < trits.Length; i++)
        {
            trits[i] = 1;
            activations[i] = 3;
        }

        var packed = TritPacking.PackLayer(trits);
        var packedStride = (50 + 4) / 5;

        var actual = TritPacking.TernaryDotPacked(
            packed.AsSpan(0, packedStride),
            activations,
            50);

        Assert.Equal(150, actual);
    }

    [Fact]
    public void TernaryDotPacked_AllNegativeTrits_NegatesActivationSum()
    {
        var trits = new sbyte[50];
        var activations = new sbyte[50];
        for (var i = 0; i < trits.Length; i++)
        {
            trits[i] = -1;
            activations[i] = 3;
        }

        var packed = TritPacking.PackLayer(trits);
        var packedStride = (50 + 4) / 5;

        var actual = TritPacking.TernaryDotPacked(
            packed.AsSpan(0, packedStride),
            activations,
            50);

        Assert.Equal(-150, actual);
    }

    [Fact]
    public void TernaryDotPacked_PaddingBytes_IgnoredBeyondTotalTrits()
    {
        // 7 trits → 2 packed bytes (1st byte holds trits 0..4, 2nd holds 5..6 + padding).
        // The padding slots must NOT contribute to the dot even if activation memory
        // beyond index 6 contains nonzero values.
        sbyte[] trits = [1, 1, 1, 1, 1, 1, 1];
        var packed = TritPacking.PackLayer(trits);
        var packedStride = (7 + 4) / 5;

        // Activations span contains 7 real values plus garbage after (activation
        // span in production only exposes totalTrits values to the method).
        sbyte[] activations = [2, 3, 4, 5, 6, 7, 8];

        var actual = TritPacking.TernaryDotPacked(
            packed.AsSpan(0, packedStride),
            activations,
            7);

        Assert.Equal(2 + 3 + 4 + 5 + 6 + 7 + 8, actual);
    }
}
