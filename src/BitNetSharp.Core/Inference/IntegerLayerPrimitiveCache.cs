using System.Runtime.CompilerServices;
using BitNetSharp.Core.Layers;

namespace BitNetSharp.Core.Inference;

/// <summary>
/// Per-layer cache of integer-inference primitives. The naive approach of
/// constructing a fresh <see cref="IntegerRmsNorm"/>, <see cref="IntegerRotaryPositionEmbedding"/>,
/// <see cref="IntegerSoftmax"/>, and <see cref="IntegerSwiGLU"/> on every
/// <see cref="IntegerForwardComposer.ForwardWithCache"/> call is dominated
/// by the RoPE sin/cos table build (O(maxSeq * headDim/2) Math.Sin/Cos
/// calls per layer per call). At Bonsai scale (36 layers, headDim=128,
/// maxSeq=128) that's ~295 k trig ops per decode step of pure overhead.
///
/// Inference weights are frozen, so we cache primitives keyed on
/// <see cref="BitNetLayer"/> identity via <see cref="ConditionalWeakTable{TKey, TValue}"/>.
/// The cache is GC-friendly: if the layer becomes unreachable the entry
/// drops with it.
///
/// RoPE rebuilds only when the requested sequence length grows past the
/// cached capacity; RmsNorms never rebuild; Softmax and SwiGLU are
/// stateless process-wide singletons.
/// </summary>
public static class IntegerLayerPrimitiveCache
{
    private sealed class PerLayer
    {
        public IntegerRmsNorm? AttnRms;
        public IntegerRmsNorm? FfnRms;
        public IntegerRotaryPositionEmbedding? Rope;
        public int RopeMaxSeq;
        public readonly object Lock = new();
    }

    private static readonly ConditionalWeakTable<BitNetLayer, PerLayer> _cache = new();
    private static readonly IntegerSoftmax _softmax = new();
    private static readonly IntegerSwiGLU _swiGLU = new();

    /// <summary>
    /// Stateless integer softmax shared across every layer and every call.
    /// </summary>
    public static IntegerSoftmax Softmax => _softmax;

    /// <summary>
    /// Stateless integer SwiGLU shared across every layer and every call.
    /// </summary>
    public static IntegerSwiGLU SwiGLU => _swiGLU;

    /// <summary>
    /// Returns the cached integer primitives for <paramref name="layer"/>,
    /// constructing them on first access. If <paramref name="ropeMaxSeq"/>
    /// exceeds the cached RoPE's capacity, the RoPE table is rebuilt to
    /// cover the new span; otherwise the existing RoPE is returned.
    /// </summary>
    public static (IntegerRmsNorm AttnRms, IntegerRmsNorm FfnRms, IntegerRotaryPositionEmbedding Rope) Get(
        BitNetLayer layer,
        int ropeMaxSeq)
    {
        ArgumentNullException.ThrowIfNull(layer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ropeMaxSeq);

        var entry = _cache.GetValue(layer, static _ => new PerLayer());

        lock (entry.Lock)
        {
            if (entry.AttnRms is null)
            {
                var attnRms = new IntegerRmsNorm(layer.Config.Dimension, layer.Config.RmsNormEpsilon);
                attnRms.ImportScale(layer.PreAttentionNorm.ExportScale());
                entry.AttnRms = attnRms;
            }

            if (entry.FfnRms is null)
            {
                var ffnRms = new IntegerRmsNorm(layer.Config.Dimension, layer.Config.RmsNormEpsilon);
                ffnRms.ImportScale(layer.PreFeedForwardNorm.ExportScale());
                entry.FfnRms = ffnRms;
            }

            if (entry.Rope is null || entry.RopeMaxSeq < ropeMaxSeq)
            {
                entry.Rope = new IntegerRotaryPositionEmbedding(
                    layer.Config.HeadDimension,
                    ropeMaxSeq,
                    layer.Config.RopeTheta);
                entry.RopeMaxSeq = ropeMaxSeq;
            }

            return (entry.AttnRms, entry.FfnRms, entry.Rope);
        }
    }
}
