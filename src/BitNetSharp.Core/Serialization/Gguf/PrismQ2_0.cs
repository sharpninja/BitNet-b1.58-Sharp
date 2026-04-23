using System;
using System.Buffers.Binary;

namespace BitNetSharp.Core.Serialization.Gguf;

/// <summary>
/// Decoder for PrismML's Q2_0 quaternary quantization (GGML_TYPE_Q2_0 = 42).
/// Block layout: FP16 scale followed by 32 bytes holding 128 two-bit codes
/// (4 quants per byte, LSB-first). Dequant rule: w = (q - 1) * d for
/// q in {0,1,2,3}, yielding quaternary {-d, 0, +d, +2d}.
/// </summary>
/// <remarks>
/// Pinned against PrismML-Eng/llama.cpp@prism (ggml.h, ggml-common.h,
/// ggml-quants.c::dequantize_row_q2_0).
/// </remarks>
public static class PrismQ2_0
{
    public const int BlockWeights = 128;
    public const int CodeBytesPerBlock = 32;
    public const int BlockBytes = 2 + CodeBytesPerBlock;
    public const uint GgmlTypeId = 42;

    public static float BlockScale(ReadOnlySpan<byte> block)
    {
        if (block.Length < BlockBytes)
        {
            throw new ArgumentException(
                $"Q2_0 block must be exactly {BlockBytes} bytes, got {block.Length}.", nameof(block));
        }

        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(block[..2]);
        return (float)BitConverter.UInt16BitsToHalf(bits);
    }

    public static float[] DequantizeBlock(ReadOnlySpan<byte> block)
    {
        var output = new float[BlockWeights];
        DequantizeBlock(block, output);
        return output;
    }

    public static void DequantizeBlock(ReadOnlySpan<byte> block, Span<float> output)
    {
        if (block.Length < BlockBytes)
        {
            throw new ArgumentException(
                $"Q2_0 block must be exactly {BlockBytes} bytes, got {block.Length}.", nameof(block));
        }
        if (output.Length < BlockWeights)
        {
            throw new ArgumentException(
                $"Output span must hold at least {BlockWeights} floats, got {output.Length}.", nameof(output));
        }

        float d = BlockScale(block);
        ReadOnlySpan<byte> codes = block.Slice(2, CodeBytesPerBlock);

        for (int j = 0; j < BlockWeights; j++)
        {
            int byteIndex = j >> 2;           // j / 4
            int bitOffset = (j & 0x3) << 1;   // (j % 4) * 2
            int q = (codes[byteIndex] >> bitOffset) & 0x3;
            output[j] = (q - 1) * d;
        }
    }

    public static (sbyte[] Trits, double WeightedAbsSum) DecodeTritsAndWeightedAbsSum(ReadOnlySpan<byte> block)
    {
        var trits = new sbyte[BlockWeights];
        DecodeTritsInto(block, trits, out double weightedAbsSum);
        return (trits, weightedAbsSum);
    }

    public static void DecodeTritsInto(ReadOnlySpan<byte> block, Span<sbyte> trits, out double weightedAbsSum)
    {
        if (block.Length < BlockBytes)
        {
            throw new ArgumentException(
                $"Q2_0 block must be exactly {BlockBytes} bytes, got {block.Length}.", nameof(block));
        }
        if (trits.Length < BlockWeights)
        {
            throw new ArgumentException(
                $"Trits span must hold at least {BlockWeights} entries, got {trits.Length}.", nameof(trits));
        }

        double d = BlockScale(block);
        ReadOnlySpan<byte> codes = block.Slice(2, CodeBytesPerBlock);

        int countQ0 = 0;
        int countQ2 = 0;
        int countQ3 = 0;

        for (int j = 0; j < BlockWeights; j++)
        {
            int byteIndex = j >> 2;
            int bitOffset = (j & 0x3) << 1;
            int q = (codes[byteIndex] >> bitOffset) & 0x3;

            // q=0 -> -1, q=1 -> 0, q=2 -> +1, q=3 -> +1 (collapse +2d magnitude onto +1 trit)
            sbyte trit;
            switch (q)
            {
                case 0: trit = -1; countQ0++; break;
                case 1: trit = 0; break;
                case 2: trit = 1; countQ2++; break;
                default: trit = 1; countQ3++; break;
            }
            trits[j] = trit;
        }

        weightedAbsSum = d * (countQ0 + countQ2 + 2 * countQ3);
    }
}
