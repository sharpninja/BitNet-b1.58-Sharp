using System;
using BitNetSharp.Distributed.Contracts;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Tests for the Phase D-4 weight blob serializer that
/// <see cref="BitNetSharp.Distributed.Coordinator.Services.WeightApplicationService"/>
/// writes to <see cref="BitNetSharp.Distributed.Coordinator.Persistence.FileSystemWeightStore"/>
/// on every accepted gradient.
/// </summary>
public sealed class WeightBlobCodecTests
{
    [Fact]
    public void Round_trips_version_and_weight_vector()
    {
        var weights = new float[] { 0.1f, -0.2f, 3.14f, -123.456f };

        var bytes = WeightBlobCodec.Encode(42L, weights);
        var decoded = WeightBlobCodec.Decode(bytes, out var version);

        Assert.Equal(42L, version);
        Assert.Equal(weights, decoded);
    }

    [Fact]
    public void Empty_vector_round_trips_with_header_only()
    {
        var bytes = WeightBlobCodec.Encode(1L, Array.Empty<float>());
        Assert.Equal(WeightBlobCodec.HeaderSize, bytes.Length);

        var decoded = WeightBlobCodec.Decode(bytes, out var version);
        Assert.Equal(1L, version);
        Assert.Empty(decoded);
    }

    [Fact]
    public void Encode_rejects_negative_version()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            WeightBlobCodec.Encode(-1L, new float[] { 1f }));
    }

    [Fact]
    public void TryDecode_rejects_short_payload()
    {
        var result = WeightBlobCodec.TryDecode(new byte[8], out _, out _, out var error);
        Assert.False(result);
        Assert.NotNull(error);
    }

    [Fact]
    public void TryDecode_rejects_bad_magic()
    {
        var bytes = new byte[WeightBlobCodec.HeaderSize];
        bytes[0] = 0xAA;
        var result = WeightBlobCodec.TryDecode(bytes, out _, out _, out var error);
        Assert.False(result);
        Assert.Contains("Magic", error);
    }

    [Fact]
    public void TryDecode_rejects_length_mismatch()
    {
        // Magic + version=1 + count=3 claimed but no float payload.
        var bytes = new byte[WeightBlobCodec.HeaderSize];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), WeightBlobCodec.Magic);
        System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(4, 8), 1L);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), 3);

        var result = WeightBlobCodec.TryDecode(bytes, out _, out _, out var error);
        Assert.False(result);
        Assert.Contains("Payload length", error);
    }
}
