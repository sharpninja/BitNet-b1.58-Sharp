using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using BitNetSharp.Core.Serialization.Gguf;

namespace BitNetSharp.Converter.Tests;

public sealed class GgufExtendedReaderTests
{
    [Fact]
    public void Read_SingleFloat32Tensor_ReturnsRawBytesAndDtype()
    {
        var tensorData = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        byte[] rawBytes = new byte[tensorData.Length * sizeof(float)];
        Buffer.BlockCopy(tensorData, 0, rawBytes, 0, rawBytes.Length);

        byte[] gguf = SyntheticGguf.Build(
            new Dictionary<string, object> { ["general.alignment"] = 32u },
            new[]
            {
                new SyntheticGguf.Tensor("tensor.f32", new[] { 4 }, GgufTensorType.F32, rawBytes),
            });

        using var stream = new MemoryStream(gguf);
        var document = GgufExtendedReader.Read(stream);

        Assert.Single(document.Tensors);
        var tensor = document.Tensors[0];
        Assert.Equal("tensor.f32", tensor.Name);
        Assert.Equal(new[] { 4 }, tensor.Dimensions);
        Assert.Equal(GgufTensorType.F32, tensor.GgmlType);
        Assert.Equal(rawBytes, tensor.RawData);
    }

    [Fact]
    public void Read_SingleFloat16Tensor_ReturnsRawBytesAndDtype()
    {
        Half[] values = { (Half)1.0f, (Half)2.0f, (Half)(-3.5f), (Half)0.25f };
        byte[] rawBytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits(values[i]);
            rawBytes[i * 2] = (byte)(bits & 0xFF);
            rawBytes[i * 2 + 1] = (byte)((bits >> 8) & 0xFF);
        }

        byte[] gguf = SyntheticGguf.Build(
            new Dictionary<string, object> { ["general.alignment"] = 32u },
            new[]
            {
                new SyntheticGguf.Tensor("tensor.f16", new[] { 4 }, GgufTensorType.F16, rawBytes),
            });

        using var stream = new MemoryStream(gguf);
        var document = GgufExtendedReader.Read(stream);

        var tensor = document.Tensors[0];
        Assert.Equal(GgufTensorType.F16, tensor.GgmlType);
        Assert.Equal(rawBytes, tensor.RawData);
    }

    [Fact]
    public void Read_SinglePrismQ2_0Tensor_ReturnsRawBlockBytes()
    {
        byte[] block = BuildQ2_0Block(0.5f);

        byte[] gguf = SyntheticGguf.Build(
            new Dictionary<string, object> { ["general.alignment"] = 32u },
            new[]
            {
                new SyntheticGguf.Tensor("tensor.q2_0", new[] { 128 }, GgufTensorType.PrismQ2_0, block),
            });

        using var stream = new MemoryStream(gguf);
        var document = GgufExtendedReader.Read(stream);

        var tensor = document.Tensors[0];
        Assert.Equal(GgufTensorType.PrismQ2_0, tensor.GgmlType);
        Assert.Equal(34, tensor.RawData.Length);
        Assert.Equal(block, tensor.RawData);
    }

    [Fact]
    public void Read_MixedTensors_EachDecodesWithCorrectDtype()
    {
        var f32Data = new float[] { 0.125f, -0.25f };
        byte[] f32Bytes = new byte[8];
        Buffer.BlockCopy(f32Data, 0, f32Bytes, 0, 8);

        byte[] f16Bytes = new byte[4];
        var half = (Half)1.5f;
        ushort halfBits = BitConverter.HalfToUInt16Bits(half);
        f16Bytes[0] = (byte)(halfBits & 0xFF);
        f16Bytes[1] = (byte)((halfBits >> 8) & 0xFF);
        halfBits = BitConverter.HalfToUInt16Bits((Half)(-2.0f));
        f16Bytes[2] = (byte)(halfBits & 0xFF);
        f16Bytes[3] = (byte)((halfBits >> 8) & 0xFF);

        byte[] q2Bytes = BuildQ2_0Block(0.25f);

        byte[] gguf = SyntheticGguf.Build(
            new Dictionary<string, object> { ["general.alignment"] = 32u },
            new[]
            {
                new SyntheticGguf.Tensor("a.f32", new[] { 2 }, GgufTensorType.F32, f32Bytes),
                new SyntheticGguf.Tensor("b.f16", new[] { 2 }, GgufTensorType.F16, f16Bytes),
                new SyntheticGguf.Tensor("c.q2_0", new[] { 128 }, GgufTensorType.PrismQ2_0, q2Bytes),
            });

        using var stream = new MemoryStream(gguf);
        var document = GgufExtendedReader.Read(stream);

        Assert.Equal(3, document.Tensors.Count);
        Assert.Equal(GgufTensorType.F32, document.Tensors[0].GgmlType);
        Assert.Equal(GgufTensorType.F16, document.Tensors[1].GgmlType);
        Assert.Equal(GgufTensorType.PrismQ2_0, document.Tensors[2].GgmlType);
        Assert.Equal(f32Bytes, document.Tensors[0].RawData);
        Assert.Equal(f16Bytes, document.Tensors[1].RawData);
        Assert.Equal(q2Bytes, document.Tensors[2].RawData);
    }

    [Fact]
    public void Read_UnsupportedTensorType_ErrorNamesOffendingTensor()
    {
        byte[] gguf = SyntheticGguf.Build(
            new Dictionary<string, object> { ["general.alignment"] = 32u },
            new[]
            {
                new SyntheticGguf.Tensor("bad.q8", new[] { 32 }, (GgufTensorType)8, new byte[36]),
            });

        using var stream = new MemoryStream(gguf);
        var ex = Assert.Throws<InvalidDataException>(() => GgufExtendedReader.Read(stream));
        Assert.Contains("bad.q8", ex.Message, StringComparison.Ordinal);
        Assert.Contains("8", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Read_StringArrayMetadata_ReturnsStringArray()
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.alignment"] = 32u,
            ["tokenizer.ggml.tokens"] = new object[] { "<s>", "</s>", "a" },
        };
        var tensorData = new float[] { 1.0f };
        byte[] rawBytes = new byte[4];
        Buffer.BlockCopy(tensorData, 0, rawBytes, 0, 4);

        byte[] gguf = SyntheticGguf.Build(metadata, new[]
        {
            new SyntheticGguf.Tensor("t", new[] { 1 }, GgufTensorType.F32, rawBytes),
        });

        using var stream = new MemoryStream(gguf);
        var document = GgufExtendedReader.Read(stream);

        Assert.True(document.Metadata.ContainsKey("tokenizer.ggml.tokens"));
        var arr = Assert.IsType<object[]>(document.Metadata["tokenizer.ggml.tokens"]);
        Assert.Equal(3, arr.Length);
        Assert.Equal("<s>", arr[0]);
        Assert.Equal("</s>", arr[1]);
        Assert.Equal("a", arr[2]);
    }

    [Fact]
    public void Read_ArchitectureAndLayerCountMetadata_AreAccessible()
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.alignment"] = 32u,
            ["general.architecture"] = "qwen3",
            ["qwen3.block_count"] = (uint)36,
            ["qwen3.embedding_length"] = (uint)4096,
        };
        var tensorData = new float[] { 1.0f };
        byte[] rawBytes = new byte[4];
        Buffer.BlockCopy(tensorData, 0, rawBytes, 0, 4);

        byte[] gguf = SyntheticGguf.Build(metadata, new[]
        {
            new SyntheticGguf.Tensor("t", new[] { 1 }, GgufTensorType.F32, rawBytes),
        });

        using var stream = new MemoryStream(gguf);
        var document = GgufExtendedReader.Read(stream);

        Assert.Equal("qwen3", document.Metadata["general.architecture"]);
        Assert.Equal(36u, document.Metadata["qwen3.block_count"]);
        Assert.Equal(4096u, document.Metadata["qwen3.embedding_length"]);
    }

    private static byte[] BuildQ2_0Block(float scale)
    {
        byte[] block = new byte[34];
        ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)scale);
        block[0] = (byte)(scaleBits & 0xFF);
        block[1] = (byte)((scaleBits >> 8) & 0xFF);
        // Pattern: q=0,1,2,3 repeating (32 bytes, 128 quants)
        for (int i = 0; i < 32; i++)
        {
            block[2 + i] = 0b_11_10_01_00;
        }
        return block;
    }
}
