using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

/// <summary>
/// Correctness harness for the SIMD-friendly 4-trit, 2-bit-signed packed
/// layout and its fused <c>TernaryDotSimdPacked</c> kernel.
///
/// This layout is the in-memory Forward-path representation; the on-disk
/// GGUF still uses the 5-trit base-3 <c>PackLayer</c> format. BitLinear
/// derives the SIMD layout once at weight import and uses it exclusively
/// during <c>Forward</c>. Both encoding round-trips and the full SIMD dot
/// must be numerically identical to the scalar oracle or 8B inference
/// will silently corrupt.
/// </summary>
public sealed class TritPackingSimdTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 5)]
    [InlineData(7, 6)]
    [InlineData(16, 7)]
    [InlineData(31, 8)]
    [InlineData(32, 9)]
    [InlineData(63, 10)]
    [InlineData(64, 11)]
    [InlineData(127, 12)]
    [InlineData(256, 13)]
    [InlineData(1024, 14)]
    [InlineData(4096, 15)]
    public void SimdPackLayer_RoundTripsViaUnpack(int totalTrits, int seed)
    {
        var random = new Random(seed);
        var trits = new sbyte[totalTrits];
        for (var i = 0; i < totalTrits; i++)
        {
            trits[i] = (sbyte)(random.Next(3) - 1);
        }

        var packed = TritPacking.SimdPackLayer(trits);
        var expectedLength = (totalTrits + 3) / 4;
        Assert.Equal(expectedLength, packed.Length);

        var unpacked = new sbyte[totalTrits];
        TritPacking.SimdUnpackLayer(packed, unpacked, totalTrits);

        for (var i = 0; i < totalTrits; i++)
        {
            Assert.Equal(trits[i], unpacked[i]);
        }
    }

    [Theory]
    [InlineData(1, 101)]
    [InlineData(2, 102)]
    [InlineData(3, 103)]
    [InlineData(4, 104)]
    [InlineData(5, 105)]
    [InlineData(7, 106)]
    [InlineData(15, 107)]
    [InlineData(16, 108)]
    [InlineData(31, 109)]
    [InlineData(32, 110)]
    [InlineData(33, 111)]
    [InlineData(63, 112)]
    [InlineData(64, 113)]
    [InlineData(127, 114)]
    [InlineData(256, 115)]
    [InlineData(1024, 116)]
    [InlineData(4096, 117)]
    public void TernaryDotSimdPacked_EqualsScalarOracle(int totalTrits, int seed)
    {
        var random = new Random(seed);
        var trits = new sbyte[totalTrits];
        var activations = new sbyte[totalTrits];
        for (var i = 0; i < totalTrits; i++)
        {
            trits[i] = (sbyte)(random.Next(3) - 1);
            activations[i] = (sbyte)(random.Next(255) - 127);
        }

        var simdPacked = TritPacking.SimdPackLayer(trits);

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

        var actual = TritPacking.TernaryDotSimdPacked(
            simdPacked,
            activations,
            totalTrits);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TernaryDotSimdPacked_AllZeroTrits_ReturnsZero()
    {
        var trits = new sbyte[100];
        var activations = new sbyte[100];
        var random = new Random(321);
        for (var i = 0; i < activations.Length; i++)
        {
            activations[i] = (sbyte)(random.Next(255) - 127);
        }

        var packed = TritPacking.SimdPackLayer(trits);
        var actual = TritPacking.TernaryDotSimdPacked(packed, activations, 100);

        Assert.Equal(0, actual);
    }

    [Fact]
    public void TernaryDotSimdPacked_AllPositiveTrits_SumsActivations()
    {
        var trits = new sbyte[50];
        var activations = new sbyte[50];
        for (var i = 0; i < trits.Length; i++)
        {
            trits[i] = 1;
            activations[i] = 3;
        }

        var packed = TritPacking.SimdPackLayer(trits);
        var actual = TritPacking.TernaryDotSimdPacked(packed, activations, 50);

        Assert.Equal(150, actual);
    }

    [Fact]
    public void TernaryDotSimdPacked_AllNegativeTrits_NegatesActivationSum()
    {
        var trits = new sbyte[50];
        var activations = new sbyte[50];
        for (var i = 0; i < trits.Length; i++)
        {
            trits[i] = -1;
            activations[i] = 3;
        }

        var packed = TritPacking.SimdPackLayer(trits);
        var actual = TritPacking.TernaryDotSimdPacked(packed, activations, 50);

        Assert.Equal(-150, actual);
    }

    [Fact]
    public void TernaryDotSimdPacked_PaddingBytes_IgnoredBeyondTotalTrits()
    {
        // 7 trits → 2 simd-packed bytes (1st byte: 4 trits, 2nd byte: 3 trits + 1 pad).
        sbyte[] trits = [1, 1, 1, 1, 1, 1, 1];
        var packed = TritPacking.SimdPackLayer(trits);

        sbyte[] activations = [2, 3, 4, 5, 6, 7, 8];

        var actual = TritPacking.TernaryDotSimdPacked(packed, activations, 7);

        Assert.Equal(2 + 3 + 4 + 5 + 6 + 7 + 8, actual);
    }

    [Fact]
    public void SimdPackLayer_VerifiesEncodingLayout()
    {
        sbyte[] trits = [1, 0, -1, 1];
        var packed = TritPacking.SimdPackLayer(trits);

        // Expected byte layout for [+1, 0, -1, +1]:
        // t0=+1 → 0b01 at bits [1:0]
        // t1= 0 → 0b00 at bits [3:2]
        // t2=-1 → 0b11 at bits [5:4]
        // t3=+1 → 0b01 at bits [7:6]
        // Byte = 0b01_11_00_01 = 0x71
        Assert.Single(packed);
        Assert.Equal(0x71, packed[0]);
    }

    [Theory]
    [InlineData(1, 201)]
    [InlineData(2, 202)]
    [InlineData(7, 203)]
    [InlineData(15, 204)]
    [InlineData(16, 205)]
    [InlineData(31, 206)]
    [InlineData(32, 207)]
    [InlineData(33, 208)]
    [InlineData(63, 209)]
    [InlineData(64, 210)]
    [InlineData(127, 211)]
    [InlineData(256, 212)]
    [InlineData(1024, 213)]
    [InlineData(4096, 214)]
    [InlineData(11008, 215)]
    public void TernaryDotSimdUnpacked_EqualsScalarOracle(int length, int seed)
    {
        var random = new Random(seed);
        var trits = new sbyte[length];
        var activations = new sbyte[length];
        for (var i = 0; i < length; i++)
        {
            trits[i] = (sbyte)(random.Next(3) - 1);
            activations[i] = (sbyte)(random.Next(255) - 127);
        }

        var expected = 0;
        for (var i = 0; i < length; i++)
        {
            var w = trits[i];
            if (w > 0) expected += activations[i];
            else if (w < 0) expected -= activations[i];
        }

        var actual = TritPacking.TernaryDotSimdUnpacked(trits, activations);
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void TernaryDotSimdUnpacked_AllZeroTrits_ReturnsZero()
    {
        var trits = new sbyte[100];
        var activations = new sbyte[100];
        var random = new Random(421);
        for (var i = 0; i < activations.Length; i++)
        {
            activations[i] = (sbyte)(random.Next(255) - 127);
        }

        Assert.Equal(0, TritPacking.TernaryDotSimdUnpacked(trits, activations));
    }

    [Fact]
    public void TernaryDotSimdUnpacked_AllPositiveTrits_SumsActivations()
    {
        var trits = new sbyte[50];
        var activations = new sbyte[50];
        for (var i = 0; i < trits.Length; i++)
        {
            trits[i] = 1;
            activations[i] = 3;
        }

        Assert.Equal(150, TritPacking.TernaryDotSimdUnpacked(trits, activations));
    }

    [Fact]
    public void TernaryDotSimdUnpacked_AllNegativeTrits_NegatesActivationSum()
    {
        var trits = new sbyte[50];
        var activations = new sbyte[50];
        for (var i = 0; i < trits.Length; i++)
        {
            trits[i] = -1;
            activations[i] = 3;
        }

        Assert.Equal(-150, TritPacking.TernaryDotSimdUnpacked(trits, activations));
    }

    [Fact]
    public void SimdUnpackLayer_AllValuesRoundTrip()
    {
        // Every possible packed byte value must decode to 4 trits in {-1, 0, +1}.
        var trits = new sbyte[4];
        for (var b = 0; b < 256; b++)
        {
            var packed = new byte[] { (byte)b };
            TritPacking.SimdUnpackLayer(packed, trits, 4);
            for (var slot = 0; slot < 4; slot++)
            {
                var expected2Bit = (b >> (slot * 2)) & 0x03;
                // 2-bit signed: 00→0, 01→+1, 10→-2 (invalid for trits), 11→-1
                var expectedTrit = expected2Bit switch
                {
                    0 => 0,
                    1 => 1,
                    2 => -2,
                    3 => -1,
                    _ => throw new InvalidOperationException()
                };
                Assert.Equal((sbyte)expectedTrit, trits[slot]);
            }
        }
    }
}
