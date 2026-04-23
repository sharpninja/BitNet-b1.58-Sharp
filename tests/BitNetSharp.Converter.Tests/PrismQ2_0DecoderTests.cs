using System;
using System.IO;
using BitNetSharp.Core.Serialization.Gguf;

namespace BitNetSharp.Converter.Tests;

public sealed class PrismQ2_0DecoderTests
{
    [Fact]
    public void Constants_MatchPrismMlForkPinnedValues()
    {
        Assert.Equal(128, PrismQ2_0.BlockWeights);
        Assert.Equal(34, PrismQ2_0.BlockBytes);
        Assert.Equal(32, PrismQ2_0.CodeBytesPerBlock);
        Assert.Equal(42u, PrismQ2_0.GgmlTypeId);
    }

    [Fact]
    public void BlockScale_ReadsFp16LittleEndianFromFirstTwoBytes()
    {
        var block = new byte[34];
        // FP16 value for 0.5 = 0x3800 little-endian -> 0x00, 0x38
        block[0] = 0x00;
        block[1] = 0x38;
        Assert.Equal(0.5f, PrismQ2_0.BlockScale(block));
    }

    [Fact]
    public void DequantizeBlock_AllFourCodes_MatchesForkReferenceOutput()
    {
        // d = 0.5, pattern repeats [q=0, q=1, q=2, q=3] for all 128 weights.
        // One byte holds 4 quants LSB-first: q0 in bits 0-1, q1 in bits 2-3, q2 in bits 4-5, q3 in bits 6-7.
        // Pattern 0,1,2,3 -> 0b11_10_01_00 = 0xE4.
        var block = new byte[34];
        WriteFp16(block, 0, 0.5f);
        for (int i = 0; i < 32; i++) { block[2 + i] = 0xE4; }

        var weights = PrismQ2_0.DequantizeBlock(block);

        Assert.Equal(128, weights.Length);
        for (int j = 0; j < 128; j++)
        {
            int q = j % 4;
            float expected = (q - 1) * 0.5f;
            Assert.Equal(expected, weights[j], 5);
        }
    }

    [Fact]
    public void DequantizeBlock_ReproducesValues_MinusD_Zero_PlusD_PlusTwoD()
    {
        var block = new byte[34];
        WriteFp16(block, 0, 1.25f);
        for (int i = 0; i < 32; i++) { block[2 + i] = 0xE4; } // same 0,1,2,3 pattern

        var weights = PrismQ2_0.DequantizeBlock(block);

        // First four weights
        Assert.Equal(-1.25f, weights[0], 5);
        Assert.Equal(0.0f, weights[1], 5);
        Assert.Equal(1.25f, weights[2], 5);
        Assert.Equal(2.5f, weights[3], 5);
    }

    [Fact]
    public void DecodeTritsAndWeightedAbsSum_CollapsesQ3ToPlusOneTrit()
    {
        var block = new byte[34];
        WriteFp16(block, 0, 0.5f);
        for (int i = 0; i < 32; i++) { block[2 + i] = 0xE4; } // 0,1,2,3 repeating

        var (trits, weightedAbsSum) = PrismQ2_0.DecodeTritsAndWeightedAbsSum(block);

        Assert.Equal(128, trits.Length);
        for (int j = 0; j < 128; j++)
        {
            int q = j % 4;
            sbyte expected = q switch
            {
                0 => (sbyte)-1,
                1 => (sbyte)0,
                2 => (sbyte)1,
                3 => (sbyte)1, // +2d collapses to +1 trit
                _ => throw new InvalidOperationException()
            };
            Assert.Equal(expected, trits[j]);
        }

        // 128 weights split evenly across q=0,1,2,3 -> 32 of each.
        // weightedAbsSum = d * (count_q0 + count_q2 + 2*count_q3)
        //               = 0.5 * (32 + 32 + 2*32) = 0.5 * 128 = 64.0
        Assert.Equal(64.0, weightedAbsSum, 5);
    }

    [Fact]
    public void DecodeTritsAndWeightedAbsSum_AllQ1_YieldsZeroTritsAndZeroSum()
    {
        var block = new byte[34];
        WriteFp16(block, 0, 2.0f);
        // All q=1 -> byte 0b01_01_01_01 = 0x55
        for (int i = 0; i < 32; i++) { block[2 + i] = 0x55; }

        var (trits, weightedAbsSum) = PrismQ2_0.DecodeTritsAndWeightedAbsSum(block);

        Assert.All(trits, t => Assert.Equal((sbyte)0, t));
        Assert.Equal(0.0, weightedAbsSum, 5);
    }

    [Fact]
    public void DecodeTritsAndWeightedAbsSum_AllQ3_YieldsAllPlusOneAndDoubledSum()
    {
        var block = new byte[34];
        WriteFp16(block, 0, 0.25f);
        // All q=3 -> byte 0b11_11_11_11 = 0xFF
        for (int i = 0; i < 32; i++) { block[2 + i] = 0xFF; }

        var (trits, weightedAbsSum) = PrismQ2_0.DecodeTritsAndWeightedAbsSum(block);

        Assert.All(trits, t => Assert.Equal((sbyte)1, t));
        // 128 weights all q=3 -> weightedAbsSum = 0.25 * (0 + 0 + 2*128) = 64.0
        Assert.Equal(64.0, weightedAbsSum, 5);
    }

    [Fact]
    public void BitLayout_IsLsbFirstFourQuantsPerByte()
    {
        // Byte 0b11_10_01_00 = 0xE4 must decode to quants [0, 1, 2, 3]
        // in positions [0, 1, 2, 3] (LSB-first).
        var block = new byte[34];
        WriteFp16(block, 0, 1.0f);
        block[2] = 0xE4;
        // Rest zero (all q=0 => all -d)

        var weights = PrismQ2_0.DequantizeBlock(block);
        Assert.Equal(-1.0f, weights[0], 5); // q=0
        Assert.Equal(0.0f, weights[1], 5);  // q=1
        Assert.Equal(1.0f, weights[2], 5);  // q=2
        Assert.Equal(2.0f, weights[3], 5);  // q=3
        // Positions 4..127 are q=0 from zero-filled bytes
        Assert.Equal(-1.0f, weights[4], 5);
        Assert.Equal(-1.0f, weights[127], 5);
    }

    [Fact]
    public void DequantizeBlock_ThrowsWhenInputTooShort()
    {
        var tooShort = new byte[33];
        Assert.Throws<ArgumentException>(() => PrismQ2_0.DequantizeBlock(tooShort));
    }

    private static void WriteFp16(byte[] dest, int offset, float value)
    {
        ushort bits = (ushort)BitConverter.HalfToUInt16Bits((Half)value);
        dest[offset] = (byte)(bits & 0xFF);
        dest[offset + 1] = (byte)((bits >> 8) & 0xFF);
    }
}
