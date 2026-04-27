namespace BitNetSharp.Core.Inference;

/// <summary>
/// Section B (KV cache quantization) - KV1: per-row absmax-quantised int8
/// K/V slab. Parallel structure to <see cref="LayerKvCache"/> but with
/// <c>sbyte</c> K/V plus a per-row <c>float</c> scale instead of fp32 K/V.
/// <para>
/// At Bonsai shape (kvDim = kvHeads * headDim = 8 * 128 = 1024, capacity 2048,
/// 36 layers): fp32 KV = 2048 * 1024 * 36 * 2 * 4 bytes ~= 576 MiB per request.
/// int8 KV ~= 144 MiB plus 144 KiB of per-row scales. The bandwidth win for
/// the FlashAttention.ForwardDecode K/V scan is 4x (1 byte/lane vs 4); the
/// dequant cost is one float-per-row scale multiply, fully amortised inside
/// the SIMD inner loop in KV3.
/// </para>
/// <para>
/// Quantisation contract matches
/// <see cref="BitNetSharp.Core.Quantization.QuantizedActivationBlock"/>:
/// per-row scale = max(|row|) / 127, with all-zero rows getting the sentinel
/// scale = 1f (so dequant of all-zero sbyte yields zero unconditionally).
/// </para>
/// </summary>
public sealed class QuantizedKvLayerCache : IKvCache
{
    private const int Int8MaxMagnitude = 127;

    public QuantizedKvLayerCache(int capacity, int kvDimension)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDimension);

        Capacity = capacity;
        KvDimension = kvDimension;
        K = new sbyte[capacity, kvDimension];
        V = new sbyte[capacity, kvDimension];
        KScale = new float[capacity];
        VScale = new float[capacity];
    }

    public int Capacity { get; }

    public int KvDimension { get; }

    public sbyte[,] K { get; }

    public sbyte[,] V { get; }

    public float[] KScale { get; }

    public float[] VScale { get; }

    /// <summary>
    /// Quantises one K row and one V row into the slot at <paramref name="row"/>.
    /// Both rows must have length = <see cref="KvDimension"/>.
    /// </summary>
    public void WriteRow(int row, ReadOnlySpan<float> kFloat, ReadOnlySpan<float> vFloat)
    {
        WriteKRow(row, kFloat);
        WriteVRow(row, vFloat);
    }

    public void WriteKRow(int row, ReadOnlySpan<float> kFloat)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Capacity);
        if (kFloat.Length != KvDimension)
        {
            throw new ArgumentException(
                $"K row length {kFloat.Length} != KvDimension {KvDimension}.", nameof(kFloat));
        }
        QuantiseRow(kFloat, K, row, KScale);
    }

    public void WriteVRow(int row, ReadOnlySpan<float> vFloat)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Capacity);
        if (vFloat.Length != KvDimension)
        {
            throw new ArgumentException(
                $"V row length {vFloat.Length} != KvDimension {KvDimension}.", nameof(vFloat));
        }
        QuantiseRow(vFloat, V, row, VScale);
    }

    /// <summary>Writes the dequantised K row into <paramref name="dst"/>.</summary>
    public void DequantizeKRow(int row, Span<float> dst) => DequantiseRow(K, row, KScale[row], dst);

    /// <summary>Writes the dequantised V row into <paramref name="dst"/>.</summary>
    public void DequantizeVRow(int row, Span<float> dst) => DequantiseRow(V, row, VScale[row], dst);

    private void QuantiseRow(ReadOnlySpan<float> src, sbyte[,] target, int row, float[] scales)
    {
        var maxAbs = 0f;
        for (var i = 0; i < src.Length; i++)
        {
            var absV = MathF.Abs(src[i]);
            if (absV > maxAbs)
            {
                maxAbs = absV;
            }
        }

        if (maxAbs <= 0f)
        {
            scales[row] = 1f;
            for (var i = 0; i < src.Length; i++)
            {
                target[row, i] = 0;
            }
            return;
        }

        var scale = maxAbs / Int8MaxMagnitude;
        scales[row] = scale;
        for (var i = 0; i < src.Length; i++)
        {
            var q = (int)MathF.Round(src[i] / scale, MidpointRounding.AwayFromZero);
            target[row, i] = (sbyte)Math.Clamp(q, -Int8MaxMagnitude, Int8MaxMagnitude);
        }
    }

    private void DequantiseRow(sbyte[,] source, int row, float scale, Span<float> dst)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Capacity);
        if (dst.Length != KvDimension)
        {
            throw new ArgumentException(
                $"Destination length {dst.Length} != KvDimension {KvDimension}.", nameof(dst));
        }

        for (var i = 0; i < KvDimension; i++)
        {
            dst[i] = source[row, i] * scale;
        }
    }
}
