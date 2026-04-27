using BitNetSharp.Core.Inference;

namespace BitNetSharp.Core.Models;

public sealed record BitNetConfig
{
    // kvHeadCount default sentinel: -1 means "derive from headCount" (matches
    // the original null-coalescing semantic). We avoid the nullable-int overload
    // because System.Text.Json requires every deserialization-constructor
    // parameter type to match a property type exactly, and KvHeadCount is
    // declared as a non-nullable int. Using -1 keeps the sentinel visible while
    // letting a single constructor serve both the ergonomic default path and
    // STJ's constructor-binding rules.
    public BitNetConfig(
        int vocabSize = 32_000,
        int dimension = 256,
        int hiddenDimension = 1_024,
        int layerCount = 4,
        int headCount = 8,
        int maxSequenceLength = 256,
        float rmsNormEpsilon = 1e-5f,
        int kvHeadCount = -1,
        float ropeTheta = 10_000f,
        KvCacheQuantization kvCacheQuantization = KvCacheQuantization.Fp32)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDimension);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);
        ArgumentOutOfRangeException.ThrowIfNegative(rmsNormEpsilon);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ropeTheta);

        if (dimension % headCount != 0)
        {
            throw new ArgumentException("The model dimension must be divisible by the head count.", nameof(dimension));
        }

        if ((dimension / headCount) % 2 != 0)
        {
            throw new ArgumentException("The per-head dimension must be even so rotary embeddings can be applied.", nameof(dimension));
        }

        int resolvedKvHeadCount = kvHeadCount < 0 ? headCount : kvHeadCount;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(resolvedKvHeadCount, nameof(kvHeadCount));

        if (headCount % resolvedKvHeadCount != 0)
        {
            throw new ArgumentException(
                $"The query head count ({headCount}) must be divisible by the KV head count ({resolvedKvHeadCount}).",
                nameof(kvHeadCount));
        }

        VocabSize = vocabSize;
        Dimension = dimension;
        HiddenDimension = hiddenDimension;
        LayerCount = layerCount;
        HeadCount = headCount;
        MaxSequenceLength = maxSequenceLength;
        RmsNormEpsilon = rmsNormEpsilon;
        KvHeadCount = resolvedKvHeadCount;
        RopeTheta = ropeTheta;
        KvCacheQuantization = kvCacheQuantization;
    }

    public int VocabSize { get; }

    public int Dimension { get; }

    public int HiddenDimension { get; }

    public int LayerCount { get; }

    public int HeadCount { get; }

    public int MaxSequenceLength { get; }

    public float RmsNormEpsilon { get; }

    public int KvHeadCount { get; }

    public float RopeTheta { get; }

    /// <summary>
    /// Section B: selects fp32 (default) or int8 K/V cache backing for
    /// cache-aware decode. <see cref="BitNetTransformer.CreateCache"/> reads
    /// this to allocate the right slab type.
    /// </summary>
    public KvCacheQuantization KvCacheQuantization { get; }

    public int HeadDimension => Dimension / HeadCount;

    public bool UsesGroupedQueryAttention => KvHeadCount < HeadCount;

    /// <summary>
    /// Qwen3-8B shape preset for importing prism-ml/Ternary-Bonsai-8B:
    /// 36 layers, dim 4096, 32 Q / 8 KV heads (GQA), hidden 12288,
    /// RoPE theta 1e6, ctx 65536. vocabSize is caller-supplied so
    /// BitNetSharp's word-level vocabulary can be kept.
    /// </summary>
    public static BitNetConfig Qwen3Like8B(int vocabSize) =>
        new(
            vocabSize: vocabSize,
            dimension: 4096,
            hiddenDimension: 12288,
            layerCount: 36,
            headCount: 32,
            maxSequenceLength: 65536,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 8,
            ropeTheta: 1_000_000f);
}
