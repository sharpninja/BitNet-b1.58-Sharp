using System.Numerics;

namespace BitNetSharp.Core.Quantization;

/// <summary>
/// Base-3 trit packing: 5 ternary weights per byte.
/// 3^5 = 243 ≤ 256, achieving ~1.6 bits/weight (within 0.5% of the
/// information-theoretic minimum of log₂(3) ≈ 1.585 bits/weight).
/// </summary>
public static class TritPacking
{
    /// <summary>
    /// LUT for decoding a packed byte into 5 ternary values {-1, 0, +1}.
    /// Index 0..242 are valid packed values; 243..255 are padding.
    /// </summary>
    public static readonly (sbyte T0, sbyte T1, sbyte T2, sbyte T3, sbyte T4)[] DecodeLut = BuildDecodeLut();

    public static byte PackFive(sbyte t0, sbyte t1, sbyte t2, sbyte t3, sbyte t4) =>
        (byte)((t0 + 1) + (t1 + 1) * 3 + (t2 + 1) * 9 + (t3 + 1) * 27 + (t4 + 1) * 81);

    public static byte[] PackLayer(sbyte[] ternaryWeights)
    {
        ArgumentNullException.ThrowIfNull(ternaryWeights);

        var total = ternaryWeights.Length;
        var packedLength = (total + 4) / 5;
        var packed = new byte[packedLength];

        for (var i = 0; i < packedLength; i++)
        {
            var baseIndex = i * 5;
            var acc = 0;
            var mul = 1;
            for (var slot = 0; slot < 5; slot++)
            {
                var wi = baseIndex + slot;
                var trit = wi < total ? ternaryWeights[wi] + 1 : 1; // pad with 0-weight (trit=1)
                acc += trit * mul;
                mul *= 3;
            }

            packed[i] = (byte)acc;
        }

        return packed;
    }

    public static sbyte[] UnpackLayer(byte[] packed, int totalWeights)
    {
        ArgumentNullException.ThrowIfNull(packed);
        ArgumentOutOfRangeException.ThrowIfNegative(totalWeights);

        var weights = new sbyte[totalWeights];

        for (var i = 0; i < packed.Length; i++)
        {
            var (t0, t1, t2, t3, t4) = DecodeLut[packed[i]];
            var baseIndex = i * 5;
            if (baseIndex < totalWeights) weights[baseIndex] = t0;
            if (baseIndex + 1 < totalWeights) weights[baseIndex + 1] = t1;
            if (baseIndex + 2 < totalWeights) weights[baseIndex + 2] = t2;
            if (baseIndex + 3 < totalWeights) weights[baseIndex + 3] = t3;
            if (baseIndex + 4 < totalWeights) weights[baseIndex + 4] = t4;
        }

        return weights;
    }

    public static void UnpackRowInto(byte[] packed, int packedOffset, int packedStride, sbyte[] buffer, int totalWeights)
    {
        for (var i = 0; i < packedStride; i++)
        {
            var (t0, t1, t2, t3, t4) = DecodeLut[packed[packedOffset + i]];
            var baseIndex = i * 5;
            if (baseIndex < totalWeights) buffer[baseIndex] = t0;
            if (baseIndex + 1 < totalWeights) buffer[baseIndex + 1] = t1;
            if (baseIndex + 2 < totalWeights) buffer[baseIndex + 2] = t2;
            if (baseIndex + 3 < totalWeights) buffer[baseIndex + 3] = t3;
            if (baseIndex + 4 < totalWeights) buffer[baseIndex + 4] = t4;
        }
    }

    /// <summary>
    /// Fused pack-native ternary dot product. Walks the packed-trit row byte-
    /// by-byte (5 trits/byte via <see cref="DecodeLut"/>), and accumulates
    /// <paramref name="activations"/> according to trit sign without ever
    /// materializing an intermediate <c>sbyte[]</c> unpack buffer.
    ///
    /// Result is numerically identical to
    /// <c>dot(UnpackLayer(packed), activations)</c> for all inputs where
    /// packed represents exactly <paramref name="totalTrits"/> trits plus
    /// zero padding in any trailing slots of the last byte.
    /// </summary>
    /// <param name="packed">
    /// Packed row; length must be <c>(totalTrits + 4) / 5</c> bytes.
    /// </param>
    /// <param name="activations">Signed int8 activations; length >= totalTrits.</param>
    /// <param name="totalTrits">Number of logical trits (ignores byte padding).</param>
    public static int TernaryDotPacked(
        ReadOnlySpan<byte> packed,
        ReadOnlySpan<sbyte> activations,
        int totalTrits)
    {
        var sum = 0;
        var fullBytes = totalTrits / 5;
        var remainder = totalTrits - fullBytes * 5;

        for (var i = 0; i < fullBytes; i++)
        {
            var (t0, t1, t2, t3, t4) = DecodeLut[packed[i]];
            var baseIndex = i * 5;
            var a0 = activations[baseIndex];
            var a1 = activations[baseIndex + 1];
            var a2 = activations[baseIndex + 2];
            var a3 = activations[baseIndex + 3];
            var a4 = activations[baseIndex + 4];
            if (t0 > 0) sum += a0; else if (t0 < 0) sum -= a0;
            if (t1 > 0) sum += a1; else if (t1 < 0) sum -= a1;
            if (t2 > 0) sum += a2; else if (t2 < 0) sum -= a2;
            if (t3 > 0) sum += a3; else if (t3 < 0) sum -= a3;
            if (t4 > 0) sum += a4; else if (t4 < 0) sum -= a4;
        }

        if (remainder > 0)
        {
            var (t0, t1, t2, t3, t4) = DecodeLut[packed[fullBytes]];
            var baseIndex = fullBytes * 5;
            if (remainder >= 1)
            {
                var a = activations[baseIndex];
                if (t0 > 0) sum += a; else if (t0 < 0) sum -= a;
            }
            if (remainder >= 2)
            {
                var a = activations[baseIndex + 1];
                if (t1 > 0) sum += a; else if (t1 < 0) sum -= a;
            }
            if (remainder >= 3)
            {
                var a = activations[baseIndex + 2];
                if (t2 > 0) sum += a; else if (t2 < 0) sum -= a;
            }
            if (remainder >= 4)
            {
                var a = activations[baseIndex + 3];
                if (t3 > 0) sum += a; else if (t3 < 0) sum -= a;
            }
        }

        return sum;
    }

    /// <summary>
    /// SIMD-friendly 4-trit-per-byte 2-bit-signed packing.
    /// Encoding: +1→0b01, 0→0b00, -1→0b11 (2-bit two's-complement signed).
    /// Slot k lives in bits [2k+1:2k] little-endian.
    /// Decode via arith right shift: <c>(sbyte)(b &lt;&lt; (6-2k)) &gt;&gt; 6</c>.
    /// ~25% larger than 5-trit base-3 but enables vpshufb-class SIMD decode.
    /// </summary>
    public static byte[] SimdPackLayer(sbyte[] ternaryWeights)
    {
        ArgumentNullException.ThrowIfNull(ternaryWeights);

        var total = ternaryWeights.Length;
        var packedLength = (total + 3) / 4;
        var packed = new byte[packedLength];

        for (var i = 0; i < packedLength; i++)
        {
            var baseIndex = i * 4;
            byte b = 0;
            for (var slot = 0; slot < 4; slot++)
            {
                var wi = baseIndex + slot;
                if (wi >= total)
                {
                    break;
                }

                var t = ternaryWeights[wi];
                // +1→0b01, 0→0b00, -1→0b11
                var code = t == 0 ? 0 : (t > 0 ? 1 : 3);
                b |= (byte)(code << (slot * 2));
            }

            packed[i] = b;
        }

        return packed;
    }

    public static void SimdUnpackLayer(ReadOnlySpan<byte> packed, Span<sbyte> trits, int totalTrits)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(totalTrits);

        for (var i = 0; i < packed.Length; i++)
        {
            var b = packed[i];
            var baseIndex = i * 4;
            for (var slot = 0; slot < 4; slot++)
            {
                var wi = baseIndex + slot;
                if (wi >= totalTrits)
                {
                    break;
                }

                // Shift slot's 2 bits to [7:6] of sbyte, then arith right shift 6 sign-extends.
                var shift = 6 - slot * 2;
                trits[wi] = (sbyte)((sbyte)(b << shift) >> 6);
            }
        }
    }

    /// <summary>
    /// Fused pack-native SIMD-friendly ternary dot. Walks 4-trits/byte
    /// packed row, decodes each slot via arith right shift (sign-extends
    /// 2-bit signed code to sbyte trit), accumulates activations with
    /// branch-free multiply (trit ∈ {-1, 0, +1} so <c>trit * act</c> is
    /// exact).
    ///
    /// When <see cref="Vector.IsHardwareAccelerated"/>, inner loop
    /// processes <see cref="Vector{sbyte}.Count"/> trits per iteration
    /// via <c>Vector.ConditionalSelect</c> over positive/negative masks,
    /// widening to <see cref="Vector{int}"/> per chunk to avoid short
    /// overflow on FFN-size rows.
    /// </summary>
    /// <summary>
    /// SIMD ternary dot against pre-decoded trits. Use this when a caller
    /// already decoded a packed row via <see cref="SimdUnpackLayer"/> and
    /// wants to amortize the decode across many activation vectors (i.e.,
    /// outer-loop-outputColumn in BitLinear.Forward).
    /// </summary>
    public static int TernaryDotSimdUnpacked(
        ReadOnlySpan<sbyte> trits,
        ReadOnlySpan<sbyte> activations)
    {
        var length = trits.Length;
        var sum = 0;
        var processed = 0;

        if (Vector.IsHardwareAccelerated && length >= Vector<sbyte>.Count)
        {
            var laneCount = Vector<sbyte>.Count;
            var vecChunks = length / laneCount;
            var sumInt = Vector<int>.Zero;
            var onesVec = Vector<sbyte>.One;
            var minusOnesVec = new Vector<sbyte>(-1);
            var zeroVec = Vector<sbyte>.Zero;

            for (var c = 0; c < vecChunks; c++)
            {
                var offset = c * laneCount;
                var tritVec = new Vector<sbyte>(trits.Slice(offset, laneCount));
                var actVec = new Vector<sbyte>(activations.Slice(offset, laneCount));
                var posMask = Vector.Equals(tritVec, onesVec);
                var negMask = Vector.Equals(tritVec, minusOnesVec);
                var pos = Vector.ConditionalSelect(posMask, actVec, zeroVec);
                var neg = Vector.ConditionalSelect(negMask, actVec, zeroVec);
                var contribution = Vector.Subtract(pos, neg);

                Vector.Widen(contribution, out Vector<short> lo, out Vector<short> hi);
                Vector.Widen(lo, out Vector<int> a, out Vector<int> b);
                Vector.Widen(hi, out Vector<int> c2, out Vector<int> d);
                sumInt = Vector.Add(sumInt, Vector.Add(Vector.Add(a, b), Vector.Add(c2, d)));
            }

            sum = Vector.Sum(sumInt);
            processed = vecChunks * laneCount;
        }

        for (var i = processed; i < length; i++)
        {
            var t = trits[i];
            if (t > 0) sum += activations[i];
            else if (t < 0) sum -= activations[i];
        }

        return sum;
    }

    public static int TernaryDotSimdPacked(
        ReadOnlySpan<byte> simdPacked,
        ReadOnlySpan<sbyte> activations,
        int totalTrits)
    {
        var sum = 0;
        var processedTrits = 0;

        if (Vector.IsHardwareAccelerated && Vector<sbyte>.Count % 4 == 0)
        {
            var laneCount = Vector<sbyte>.Count;
            var bytesPerChunk = laneCount / 4;
            var chunks = totalTrits / laneCount;

            if (chunks > 0)
            {
                Span<sbyte> tritBuffer = stackalloc sbyte[laneCount];
                var onesVec = Vector<sbyte>.One;
                var minusOnesVec = new Vector<sbyte>(-1);
                var zeroVec = Vector<sbyte>.Zero;
                var sumInt = Vector<int>.Zero;

                for (var c = 0; c < chunks; c++)
                {
                    var packedOffset = c * bytesPerChunk;
                    var actOffset = c * laneCount;

                    // Decode chunk: 2-bit signed → sbyte.
                    for (var j = 0; j < bytesPerChunk; j++)
                    {
                        var b = simdPacked[packedOffset + j];
                        var baseSlot = j * 4;
                        tritBuffer[baseSlot] = (sbyte)((sbyte)(b << 6) >> 6);
                        tritBuffer[baseSlot + 1] = (sbyte)((sbyte)(b << 4) >> 6);
                        tritBuffer[baseSlot + 2] = (sbyte)((sbyte)(b << 2) >> 6);
                        tritBuffer[baseSlot + 3] = (sbyte)((sbyte)b >> 6);
                    }

                    var trits = new Vector<sbyte>(tritBuffer);
                    var acts = new Vector<sbyte>(activations.Slice(actOffset, laneCount));
                    var posMask = Vector.Equals(trits, onesVec);
                    var negMask = Vector.Equals(trits, minusOnesVec);
                    var pos = Vector.ConditionalSelect(posMask, acts, zeroVec);
                    var neg = Vector.ConditionalSelect(negMask, acts, zeroVec);
                    var contribution = Vector.Subtract(pos, neg);

                    Vector.Widen(contribution, out Vector<short> lo, out Vector<short> hi);
                    Vector.Widen(lo, out Vector<int> a, out Vector<int> b2);
                    Vector.Widen(hi, out Vector<int> c2, out Vector<int> d);
                    sumInt = Vector.Add(sumInt, Vector.Add(Vector.Add(a, b2), Vector.Add(c2, d)));
                }

                sum = Vector.Sum(sumInt);
                processedTrits = chunks * laneCount;
            }
        }

        // Scalar tail: handle leftover full bytes + partial byte.
        var fullBytes = totalTrits / 4;
        var startByte = processedTrits / 4;

        for (var i = startByte; i < fullBytes; i++)
        {
            var b = simdPacked[i];
            var baseIndex = i * 4;
            sbyte t0 = (sbyte)((sbyte)(b << 6) >> 6);
            sbyte t1 = (sbyte)((sbyte)(b << 4) >> 6);
            sbyte t2 = (sbyte)((sbyte)(b << 2) >> 6);
            sbyte t3 = (sbyte)((sbyte)b >> 6);
            sum += t0 * activations[baseIndex];
            sum += t1 * activations[baseIndex + 1];
            sum += t2 * activations[baseIndex + 2];
            sum += t3 * activations[baseIndex + 3];
        }

        var remainder = totalTrits - fullBytes * 4;
        if (remainder > 0)
        {
            var b = simdPacked[fullBytes];
            var baseIndex = fullBytes * 4;
            if (remainder >= 1)
            {
                sbyte t = (sbyte)((sbyte)(b << 6) >> 6);
                sum += t * activations[baseIndex];
            }
            if (remainder >= 2)
            {
                sbyte t = (sbyte)((sbyte)(b << 4) >> 6);
                sum += t * activations[baseIndex + 1];
            }
            if (remainder >= 3)
            {
                sbyte t = (sbyte)((sbyte)(b << 2) >> 6);
                sum += t * activations[baseIndex + 2];
            }
        }

        return sum;
    }

    private static (sbyte, sbyte, sbyte, sbyte, sbyte)[] BuildDecodeLut()
    {
        var lut = new (sbyte, sbyte, sbyte, sbyte, sbyte)[256];
        for (var b = 0; b < 256; b++)
        {
            var v = b;
            var t0 = (sbyte)(v % 3 - 1); v /= 3;
            var t1 = (sbyte)(v % 3 - 1); v /= 3;
            var t2 = (sbyte)(v % 3 - 1); v /= 3;
            var t3 = (sbyte)(v % 3 - 1); v /= 3;
            var t4 = (sbyte)(v % 3 - 1);
            lut[b] = (t0, t1, t2, t3, t4);
        }

        return lut;
    }
}
