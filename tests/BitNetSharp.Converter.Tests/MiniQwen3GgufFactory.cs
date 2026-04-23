using System;
using System.Collections.Generic;
using BitNetSharp.Core.Serialization.Gguf;

namespace BitNetSharp.Converter.Tests;

/// <summary>
/// Produces a synthetic Qwen3-architecture Prism-Q2_0 GGUF byte stream that
/// matches the shape expected by <c>BitNetConfig.Qwen3Like8B</c> at small
/// dimensions. Shared across Qwen3BonsaiConverterTests and ImportGgufCommandTests.
/// </summary>
internal static class MiniQwen3GgufFactory
{
    public const int Dim = 64;
    public const int HeadCount = 4;
    public const int KvHeadCount = 2;
    public const int HeadDim = Dim / HeadCount; // 16
    public const int KvDim = KvHeadCount * HeadDim; // 32
    public const int Hidden = 128;
    public const int LayerCount = 2;

    public static byte[] Build(
        string architecture = "qwen3",
        float q2Scale = 0.5f,
        byte q2Pattern = 0xE4,
        float normScaleValue = 1.25f,
        int? overrideLayerCount = null)
    {
        int layers = overrideLayerCount ?? LayerCount;
        var metadata = new Dictionary<string, object>
        {
            ["general.alignment"] = 32u,
            ["general.architecture"] = architecture,
            ["qwen3.block_count"] = (uint)layers,
            ["qwen3.embedding_length"] = (uint)Dim,
            ["qwen3.attention.head_count"] = (uint)HeadCount,
            ["qwen3.attention.head_count_kv"] = (uint)KvHeadCount,
            ["qwen3.feed_forward_length"] = (uint)Hidden,
        };
        var tensors = BuildBodyTensors(q2Scale, q2Pattern, normScaleValue, layers);
        return SyntheticGguf.Build(metadata, tensors);
    }

    public static List<SyntheticGguf.Tensor> BuildBodyTensors(
        float q2Scale, byte q2Pattern, float normScaleValue, int layers)
    {
        var tensors = new List<SyntheticGguf.Tensor>
        {
            new("output_norm.weight", new[] { Dim }, GgufTensorType.F16, BuildF16NormScale(Dim, normScaleValue)),
            // token_embd.weight and output.weight: converter discards these.
            new("token_embd.weight", new[] { Dim, 8 }, GgufTensorType.F16, new byte[Dim * 8 * 2]),
            new("output.weight", new[] { Dim, 8 }, GgufTensorType.F16, new byte[Dim * 8 * 2]),
        };

        for (int i = 0; i < layers; i++)
        {
            string p = $"blk.{i}.";
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_norm.weight", new[] { Dim }, GgufTensorType.F16, BuildF16NormScale(Dim, normScaleValue)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_q.weight", new[] { Dim, Dim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Dim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_k.weight", new[] { Dim, KvDim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * KvDim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_v.weight", new[] { Dim, KvDim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * KvDim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_output.weight", new[] { Dim, Dim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Dim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_norm.weight", new[] { Dim }, GgufTensorType.F16, BuildF16NormScale(Dim, normScaleValue)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_gate.weight", new[] { Dim, Hidden }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Hidden, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_up.weight", new[] { Dim, Hidden }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Hidden, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_down.weight", new[] { Hidden, Dim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Hidden * Dim, q2Scale, q2Pattern)));
        }

        return tensors;
    }

    public static byte[] BuildF16NormScale(int dim, float value)
    {
        byte[] bytes = new byte[dim * 2];
        ushort halfBits = BitConverter.HalfToUInt16Bits((Half)value);
        for (int i = 0; i < dim; i++)
        {
            bytes[i * 2] = (byte)(halfBits & 0xFF);
            bytes[i * 2 + 1] = (byte)((halfBits >> 8) & 0xFF);
        }
        return bytes;
    }

    public static byte[] BuildQ2Bytes(int elementCount, float scale, byte codePattern)
    {
        int blocks = elementCount / 128;
        byte[] bytes = new byte[blocks * 34];
        ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)scale);
        for (int b = 0; b < blocks; b++)
        {
            int off = b * 34;
            bytes[off] = (byte)(scaleBits & 0xFF);
            bytes[off + 1] = (byte)((scaleBits >> 8) & 0xFF);
            for (int j = 0; j < 32; j++)
            {
                bytes[off + 2 + j] = codePattern;
            }
        }
        return bytes;
    }
}
