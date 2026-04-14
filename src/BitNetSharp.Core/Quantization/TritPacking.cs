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
