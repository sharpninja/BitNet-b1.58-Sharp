using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

public sealed class TritPackingTests
{
    [Fact]
    public void PackUnpack_RoundTrips()
    {
        sbyte[] trits = [1, -1, 0, 1, -1];
        var packed = TritPacking.PackFive(trits[0], trits[1], trits[2], trits[3], trits[4]);
        var (t0, t1, t2, t3, t4) = TritPacking.DecodeLut[packed];

        Assert.Equal(trits[0], t0);
        Assert.Equal(trits[1], t1);
        Assert.Equal(trits[2], t2);
        Assert.Equal(trits[3], t3);
        Assert.Equal(trits[4], t4);
    }

    [Fact]
    public void AllCombinations_RoundTrip()
    {
        // Exhaustively test all 243 valid combinations
        var count = 0;
        for (sbyte a = -1; a <= 1; a++)
        for (sbyte b = -1; b <= 1; b++)
        for (sbyte c = -1; c <= 1; c++)
        for (sbyte d = -1; d <= 1; d++)
        for (sbyte e = -1; e <= 1; e++)
        {
            var packed = TritPacking.PackFive(a, b, c, d, e);
            Assert.True(packed <= 242, $"Packed value {packed} exceeds 242 for ({a},{b},{c},{d},{e})");

            var (t0, t1, t2, t3, t4) = TritPacking.DecodeLut[packed];
            Assert.Equal(a, t0);
            Assert.Equal(b, t1);
            Assert.Equal(c, t2);
            Assert.Equal(d, t3);
            Assert.Equal(e, t4);
            count++;
        }

        Assert.Equal(243, count);
    }

    [Theory]
    [InlineData(5, 1)]
    [InlineData(10, 2)]
    [InlineData(7, 2)]
    [InlineData(11, 3)]
    [InlineData(1, 1)]
    [InlineData(0, 0)]
    public void PackArray_CorrectLength(int totalWeights, int expectedPackedLength)
    {
        var weights = new sbyte[totalWeights];
        var packed = TritPacking.PackLayer(weights);
        Assert.Equal(expectedPackedLength, packed.Length);
    }

    [Theory]
    [InlineData(7)]
    [InlineData(11)]
    [InlineData(13)]
    [InlineData(3)]
    public void NonMultipleOfFive_PadsCorrectly(int totalWeights)
    {
        var weights = new sbyte[totalWeights];
        for (var i = 0; i < totalWeights; i++)
            weights[i] = (sbyte)((i % 3) - 1); // -1, 0, 1, -1, 0, ...

        var packed = TritPacking.PackLayer(weights);
        var unpacked = TritPacking.UnpackLayer(packed, totalWeights);

        Assert.Equal(totalWeights, unpacked.Length);
        for (var i = 0; i < totalWeights; i++)
            Assert.Equal(weights[i], unpacked[i]);
    }

    [Fact]
    public void PackUnpackLayer_LargeArray_RoundTrips()
    {
        var random = new Random(42);
        var weights = new sbyte[1024];
        for (var i = 0; i < weights.Length; i++)
            weights[i] = (sbyte)(random.Next(3) - 1);

        var packed = TritPacking.PackLayer(weights);
        var unpacked = TritPacking.UnpackLayer(packed, weights.Length);

        Assert.Equal(weights.Length, unpacked.Length);
        for (var i = 0; i < weights.Length; i++)
            Assert.Equal(weights[i], unpacked[i]);
    }

    [Fact]
    public void UnpackRowInto_MatchesUnpackLayer()
    {
        var random = new Random(42);
        var weights = new sbyte[100];
        for (var i = 0; i < weights.Length; i++)
            weights[i] = (sbyte)(random.Next(3) - 1);

        var packed = TritPacking.PackLayer(weights);
        var packedStride = (100 + 4) / 5;
        var buffer = new sbyte[100];

        TritPacking.UnpackRowInto(packed, 0, packedStride, buffer, 100);

        for (var i = 0; i < 100; i++)
            Assert.Equal(weights[i], buffer[i]);
    }
}
