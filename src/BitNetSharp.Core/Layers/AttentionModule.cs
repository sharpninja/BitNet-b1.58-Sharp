using BitNetSharp.Core.Inference;

namespace BitNetSharp.Core.Layers;

/// <summary>
/// Common base for attention layers (MultiHeadAttention, GroupedQueryAttention).
/// Exposes the four BitLinear projections so trainers, audit passes, and model
/// parameter iteration can treat both attention flavors uniformly.
/// </summary>
public abstract class AttentionModule : Module
{
    public abstract BitLinear QueryProjection { get; }

    public abstract BitLinear KeyProjection { get; }

    public abstract BitLinear ValueProjection { get; }

    public abstract BitLinear OutputProjection { get; }

    public abstract float AttentionScale { get; }

    public virtual bool UsesRotaryPositionEmbedding => true;

    public virtual bool AppliesRotaryPositionEmbeddingToQueriesAndKeysOnly => true;

    public virtual bool UsesCausalAttentionMask => true;

    public abstract long EstimateResidentParameterBytes();

    /// <summary>
    /// Cache-aware forward for inference. Processes rows of <paramref name="input"/>
    /// as positions [positionOffset, positionOffset + input.Rows), appending new
    /// K/V rows into <paramref name="cache"/>, and attends against all cached K/V
    /// rows [0, positionOffset + input.Rows). Default implementation throws; each
    /// attention flavor provides its own override.
    /// </summary>
    public virtual float[,] Forward(float[,] input, LayerKvCache cache, int positionOffset)
        => throw new NotSupportedException($"{GetType().Name} does not implement cache-aware Forward.");

    /// <summary>
    /// Fused flash-style single-row decode. Skips materialising the
    /// [headCount, 1, pastLength] weights tensor that the standard dense
    /// path produces. Input must be exactly one row (query length == 1);
    /// use <see cref="Forward(float[,], LayerKvCache, int)"/> for prefill.
    /// </summary>
    public virtual float[,] ForwardFlashDecode(float[,] input, LayerKvCache cache, int positionOffset)
        => throw new NotSupportedException($"{GetType().Name} does not implement flash decode.");

    /// <summary>
    /// Section B (KV cache quantization) - KV5: cache-aware forward with
    /// int8 K/V backing. Same contract as
    /// <see cref="Forward(float[,], LayerKvCache, int)"/> but writes
    /// per-row absmax-quantised K/V into <paramref name="cache"/> and
    /// dispatches the dot-side path through the int8 SIMD kernels.
    /// </summary>
    public virtual float[,] Forward(float[,] input, QuantizedKvLayerCache cache, int positionOffset)
        => throw new NotSupportedException($"{GetType().Name} does not implement int8 cache-aware Forward.");

    /// <summary>
    /// Section B (KV cache quantization) - KV5: flash decode with int8 K/V
    /// backing. Same contract as
    /// <see cref="ForwardFlashDecode(float[,], LayerKvCache, int)"/> but
    /// uses <see cref="QuantizedKvLayerCache"/> + the int8 online-softmax
    /// kernel for the per-head attention pass.
    /// </summary>
    public virtual float[,] ForwardFlashDecode(float[,] input, QuantizedKvLayerCache cache, int positionOffset)
        => throw new NotSupportedException($"{GetType().Name} does not implement int8 flash decode.");
}
