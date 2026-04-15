using System;
using System.Buffers.Binary;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Wire-format codec for the global weight vector blobs the
/// coordinator persists to its weight store and workers download
/// via <c>GET /weights/{version}</c>. Phase D-4 uses a simple
/// dense float32 vector; the blob format is stable so later phases
/// can extend it without breaking on-disk compatibility.
///
/// <para>
/// Binary layout (little-endian, no padding):
/// <code>
///   u32  magic   = 0x54475742 ("BGWT" in little-endian ASCII)
///   i64  version
///   i32  count
///   f32[] values (count elements)
/// </code>
/// Total size: 16 + 4 * count bytes.
/// </para>
/// </summary>
public static class WeightBlobCodec
{
    /// <summary>Magic bytes in the weight blob header.</summary>
    public const uint Magic = 0x54475742u;

    /// <summary>Header size in bytes: magic + version + count.</summary>
    public const int HeaderSize = 16;

    /// <summary>
    /// Serializes a weight vector + version to the wire format.
    /// </summary>
    public static byte[] Encode(long version, ReadOnlySpan<float> weights)
    {
        if (version < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(version), "Version must be non-negative.");
        }

        var count = weights.Length;
        var bytes = new byte[HeaderSize + 4 * count];
        var span = bytes.AsSpan();

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], Magic);
        BinaryPrimitives.WriteInt64LittleEndian(span.Slice(4, 8), version);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(12, 4), count);

        var floats = span[HeaderSize..];
        for (var i = 0; i < count; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(floats.Slice(i * 4, 4), weights[i]);
        }

        return bytes;
    }

    /// <summary>
    /// Deserializes a wire blob into <paramref name="version"/> +
    /// a fresh float array. Throws on malformed input.
    /// </summary>
    public static float[] Decode(ReadOnlySpan<byte> payload, out long version)
    {
        if (!TryDecode(payload, out version, out var weights, out var error))
        {
            throw new ArgumentException($"Weight blob could not be decoded: {error}", nameof(payload));
        }

        return weights;
    }

    /// <summary>
    /// Attempts to deserialize the blob. Returns <c>false</c> + a
    /// user-readable <paramref name="error"/> instead of throwing.
    /// </summary>
    public static bool TryDecode(
        ReadOnlySpan<byte> payload,
        out long version,
        out float[] weights,
        out string? error)
    {
        version = 0;
        weights = Array.Empty<float>();

        if (payload.Length < HeaderSize)
        {
            error = $"Payload is {payload.Length} bytes; need at least {HeaderSize}.";
            return false;
        }

        var magic = BinaryPrimitives.ReadUInt32LittleEndian(payload[..4]);
        if (magic != Magic)
        {
            error = $"Magic 0x{magic:X8} does not match 0x{Magic:X8}.";
            return false;
        }

        version = BinaryPrimitives.ReadInt64LittleEndian(payload.Slice(4, 8));
        var count = BinaryPrimitives.ReadInt32LittleEndian(payload.Slice(12, 4));
        if (count < 0)
        {
            error = $"Negative count {count}.";
            return false;
        }

        var expected = HeaderSize + 4 * count;
        if (payload.Length != expected)
        {
            error = $"Payload length {payload.Length} != expected {expected} for count {count}.";
            return false;
        }

        var floats = new float[count];
        var bytes = payload[HeaderSize..];
        for (var i = 0; i < count; i++)
        {
            floats[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice(i * 4, 4));
        }

        weights = floats;
        error = null;
        return true;
    }
}
