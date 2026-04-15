using System;
using System.Linq;
using BitNetSharp.Distributed.Contracts;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for the Phase D-4 int8 + per-tensor scale
/// gradient codec. Pins the binary format, quantization error
/// bound, error-feedback residual semantics, and the
/// TryDecode sad-path coverage.
/// </summary>
public sealed class Int8GradientCodecTests
{
    [Fact]
    public void Encode_empty_gradient_produces_header_only_with_zero_scale()
    {
        var residual = Array.Empty<float>();
        var bytes = Int8GradientCodec.Encode(Array.Empty<float>(), residual);

        Assert.Equal(Int8GradientCodec.HeaderSize, bytes.Length);
        Assert.True(Int8GradientCodec.TryDecode(bytes, out var decoded, out var error));
        Assert.Null(error);
        Assert.Empty(decoded);
    }

    [Fact]
    public void Encode_all_zeros_emits_zero_scale_and_zero_bytes()
    {
        var gradient = new float[16];
        var residual = new float[16];

        var bytes = Int8GradientCodec.Encode(gradient, residual);

        Assert.Equal(Int8GradientCodec.HeaderSize + 16, bytes.Length);
        Assert.All(residual, r => Assert.Equal(0f, r));
        var decoded = Int8GradientCodec.Decode(bytes);
        Assert.All(decoded, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Encode_then_decode_round_trip_is_within_half_scale_error()
    {
        var rng = new Random(42);
        var gradient = new float[128];
        for (var i = 0; i < gradient.Length; i++)
        {
            gradient[i] = (float)((rng.NextDouble() - 0.5) * 2.0); // [-1, 1]
        }
        var residual = new float[gradient.Length];

        var bytes = Int8GradientCodec.Encode(gradient, residual);
        var decoded = Int8GradientCodec.Decode(bytes);

        // Scale is max(abs)/127 so half-scale error per element.
        var absMax = gradient.Select(MathF.Abs).Max();
        var scale = absMax / 127f;
        var tolerance = scale / 2f + 1e-6f;

        Assert.Equal(gradient.Length, decoded.Length);
        for (var i = 0; i < gradient.Length; i++)
        {
            Assert.InRange(decoded[i], gradient[i] - tolerance, gradient[i] + tolerance);
        }
    }

    [Fact]
    public void Residual_plus_decoded_equals_original_gradient()
    {
        var gradient = new float[] { 0.31f, -0.77f, 1.2f, 0f, 0.01f };
        var residual = new float[gradient.Length];

        var bytes = Int8GradientCodec.Encode(gradient, residual);
        var decoded = Int8GradientCodec.Decode(bytes);

        for (var i = 0; i < gradient.Length; i++)
        {
            Assert.Equal(gradient[i], decoded[i] + residual[i], precision: 4);
        }
    }

    [Fact]
    public void Encoder_clamps_values_that_would_overflow_int8()
    {
        // Construct a gradient where max abs dominates. Everything
        // stays within [-127, 127] after scaling by design; this
        // test just asserts that Decode reproduces the extreme
        // element exactly and the rest fall into the quant grid.
        var gradient = new float[] { -10f, 0f, 10f };
        var residual = new float[3];
        var bytes = Int8GradientCodec.Encode(gradient, residual);
        var decoded = Int8GradientCodec.Decode(bytes);

        Assert.Equal(-10f, decoded[0], precision: 3);
        Assert.Equal(10f, decoded[2], precision: 3);
        Assert.Equal(0f, decoded[1]);
    }

    [Fact]
    public void TryDecode_rejects_too_short_payload()
    {
        var result = Int8GradientCodec.TryDecode(new byte[6], out var gradient, out var error);
        Assert.False(result);
        Assert.Empty(gradient);
        Assert.NotNull(error);
    }

    [Fact]
    public void TryDecode_rejects_bad_magic()
    {
        var bad = new byte[Int8GradientCodec.HeaderSize];
        bad[0] = 0xFF;
        var result = Int8GradientCodec.TryDecode(bad, out var gradient, out var error);
        Assert.False(result);
        Assert.Empty(gradient);
        Assert.Contains("Magic", error);
    }

    [Fact]
    public void TryDecode_rejects_length_mismatch()
    {
        // Magic + count=5 + scale=1 but no payload bytes.
        var header = new byte[Int8GradientCodec.HeaderSize];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0, 4), Int8GradientCodec.Magic);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(4, 4), 5);
        System.Buffers.Binary.BinaryPrimitives.WriteSingleLittleEndian(header.AsSpan(8, 4), 1f);

        var result = Int8GradientCodec.TryDecode(header, out _, out var error);
        Assert.False(result);
        Assert.Contains("Payload length", error);
    }
}
