using System;
using System.Buffers.Binary;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Wire-format codec for per-tensor 8-bit integer gradient quantization
/// with a single float32 scale. Used by the Phase D-4 gradient path:
/// workers encode their float32 gradient vector, ship it in
/// <see cref="GradientSubmission.GradientPayload"/> with
/// <see cref="GradientSubmission.GradientFormat"/> = <see cref="FormatId"/>,
/// and the coordinator decodes it back to float32 before applying to
/// the global weight vector.
///
/// <para>
/// Binary layout (little-endian, no padding):
/// <code>
///   u32  magic       = 0x46454742 ("BGEF" in little-endian ASCII)
///   u32  count       = element count
///   f32  scale       = per-tensor scale
///   i8[] values      = count quantized bytes, clamped to [-127, 127]
/// </code>
/// Total size: 12 + count bytes. A zero-length gradient is valid and
/// encodes as the 12-byte header alone (scale = 0).
/// </para>
///
/// <para>
/// Quantization policy: the encoder computes <c>scale = max(abs(g)) / 127</c>
/// and then rounds each element to the nearest integer in
/// <c>[-127, 127]</c>. The decoder reverses the map as
/// <c>g = value * scale</c>. Round-trip error is bounded by
/// <c>scale/2</c> per element, and the encoder also returns the
/// per-element residual so the caller can feed it forward into the
/// next step (error feedback) for bias correction.
/// </para>
/// </summary>
public static class Int8GradientCodec
{
    /// <summary>Format identifier stored on <see cref="GradientSubmission"/>.</summary>
    public const string FormatId = "int8-ef-v1";

    /// <summary>Magic bytes in the blob header: BGEF (little-endian).</summary>
    public const uint Magic = 0x46454742u;

    /// <summary>Header size in bytes: magic + count + scale.</summary>
    public const int HeaderSize = 12;

    /// <summary>
    /// Encodes <paramref name="gradient"/> using int8 + float32 scale
    /// quantization and writes the per-element residual into
    /// <paramref name="residual"/> (must be the same length as the
    /// gradient). The residual is what error-feedback callers add
    /// into the next step's gradient before encoding.
    /// </summary>
    public static byte[] Encode(ReadOnlySpan<float> gradient, Span<float> residual)
    {
        if (residual.Length != gradient.Length)
        {
            throw new ArgumentException(
                $"residual length ({residual.Length}) must match gradient length ({gradient.Length}).",
                nameof(residual));
        }

        var count = gradient.Length;
        var bytes = new byte[HeaderSize + count];
        var span = bytes.AsSpan();

        BinaryPrimitives.WriteUInt32LittleEndian(span[..4], Magic);
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(4, 4), count);

        if (count == 0)
        {
            BinaryPrimitives.WriteSingleLittleEndian(span.Slice(8, 4), 0f);
            return bytes;
        }

        // Find absolute maximum for the per-tensor scale.
        var absMax = 0f;
        for (var i = 0; i < count; i++)
        {
            var abs = MathF.Abs(gradient[i]);
            if (abs > absMax)
            {
                absMax = abs;
            }
        }

        var scale = absMax > 0f ? absMax / 127f : 0f;
        BinaryPrimitives.WriteSingleLittleEndian(span.Slice(8, 4), scale);

        var values = span[HeaderSize..];
        if (scale <= 0f)
        {
            // All zeros. Emit an all-zero value array and clear residuals.
            for (var i = 0; i < count; i++)
            {
                values[i] = 0;
                residual[i] = 0f;
            }

            return bytes;
        }

        var invScale = 1f / scale;
        for (var i = 0; i < count; i++)
        {
            var scaled = gradient[i] * invScale;
            var rounded = MathF.Round(scaled, MidpointRounding.AwayFromZero);
            var clamped = Math.Clamp((int)rounded, -127, 127);
            values[i] = (byte)(sbyte)clamped;
            residual[i] = gradient[i] - clamped * scale;
        }

        return bytes;
    }

    /// <summary>
    /// Decodes a blob produced by <see cref="Encode"/> into a fresh
    /// <c>float[]</c>. Throws on malformed input.
    /// </summary>
    public static float[] Decode(ReadOnlySpan<byte> payload)
    {
        if (!TryDecode(payload, out var result, out var error))
        {
            throw new ArgumentException($"Gradient payload could not be decoded: {error}", nameof(payload));
        }

        return result;
    }

    /// <summary>
    /// Attempts to decode the blob. Returns <c>false</c> + a
    /// user-readable <paramref name="error"/> message instead of
    /// throwing so callers can map the failure to an HTTP 400.
    /// </summary>
    public static bool TryDecode(ReadOnlySpan<byte> payload, out float[] gradient, out string? error)
    {
        if (payload.Length < HeaderSize)
        {
            gradient = Array.Empty<float>();
            error = $"Payload is {payload.Length} bytes; need at least {HeaderSize}.";
            return false;
        }

        var magic = BinaryPrimitives.ReadUInt32LittleEndian(payload[..4]);
        if (magic != Magic)
        {
            gradient = Array.Empty<float>();
            error = $"Magic 0x{magic:X8} does not match 0x{Magic:X8}.";
            return false;
        }

        var count = BinaryPrimitives.ReadInt32LittleEndian(payload.Slice(4, 4));
        if (count < 0)
        {
            gradient = Array.Empty<float>();
            error = $"Negative element count {count}.";
            return false;
        }

        if (payload.Length != HeaderSize + count)
        {
            gradient = Array.Empty<float>();
            error = $"Payload length {payload.Length} != header+{count} ({HeaderSize + count}).";
            return false;
        }

        var scale = BinaryPrimitives.ReadSingleLittleEndian(payload.Slice(8, 4));
        if (!float.IsFinite(scale) || scale < 0f)
        {
            gradient = Array.Empty<float>();
            error = $"Non-finite or negative scale {scale}.";
            return false;
        }

        var values = payload[HeaderSize..];
        var output = new float[count];
        for (var i = 0; i < count; i++)
        {
            var quant = (sbyte)values[i];
            output[i] = quant * scale;
        }

        gradient = output;
        error = null;
        return true;
    }
}
