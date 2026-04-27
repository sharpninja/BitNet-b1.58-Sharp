namespace BitNetSharp.Core.Inference;

/// <summary>
/// Selects the K/V cache element type for cache-aware decode. Set on
/// <see cref="BitNetSharp.Core.Models.BitNetConfig.KvCacheQuantization"/>;
/// consumed by <see cref="BitNetSharp.Core.Models.BitNetTransformer.CreateCache"/>.
/// </summary>
public enum KvCacheQuantization
{
    /// <summary>
    /// Default: <see cref="LayerKvCache"/> with fp32 K/V (4 bytes/lane).
    /// Bit-exact across all attention paths; matches pre-Section-B behaviour.
    /// </summary>
    Fp32 = 0,

    /// <summary>
    /// Section B: <see cref="QuantizedKvLayerCache"/> with int8 K/V plus
    /// per-row absmax fp32 scale (1 byte/lane + ~0.4% scale tax). Bandwidth
    /// win compounds with layer count and request capacity. Tradeoff: per-row
    /// absmax quantisation introduces small numerical error vs the fp32 path
    /// (target relative error <= 5e-2 end-to-end on small models).
    /// </summary>
    Int8 = 1,
}
