namespace BitNetSharp.Core.Inference;

/// <summary>
/// Section B (KV cache quantization) - KV2: polymorphic write contract that
/// both <see cref="LayerKvCache"/> (fp32 K/V) and
/// <see cref="QuantizedKvLayerCache"/> (int8 K/V with per-row scale)
/// implement. Cache-aware attention (MHA / GQA) writes through this
/// interface so the same call site handles either backing.
/// <para>
/// The dot-side path (KV3 / KV4) does not go through this interface; the
/// AttentionMath.DotInt8 / FlashAttention.ForwardDecodeInt8 kernels take
/// the raw sbyte spans + scale arrays directly so the SIMD inner loop
/// stays branch-free. The cache type is checked once at the top of the
/// attention forward and dispatched to the right kernel.
/// </para>
/// </summary>
public interface IKvCache
{
    int Capacity { get; }

    int KvDimension { get; }

    /// <summary>
    /// Writes one K row into the slot at <paramref name="row"/>. For fp32 the
    /// write is exact; for int8 the row is quantised against its own absmax.
    /// </summary>
    void WriteKRow(int row, ReadOnlySpan<float> kFloat);

    /// <summary>
    /// Writes one V row into the slot at <paramref name="row"/>. For fp32 the
    /// write is exact; for int8 the row is quantised against its own absmax.
    /// </summary>
    void WriteVRow(int row, ReadOnlySpan<float> vFloat);
}
