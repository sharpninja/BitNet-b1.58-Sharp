namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Named model-architecture presets for the Truck Mate intent-
/// classification SLM. Each preset targets a different point on
/// the quality / latency / memory trade-off curve.
///
/// <para>
/// Parameter counts assume VocabSize = 5174 (the word-level
/// tokenizer trained on the 50K Truck Mate synthetic corpus).
/// Actual counts will be slightly different if the vocab size
/// changes after retraining the tokenizer on a larger corpus.
/// </para>
///
/// <para>
/// All presets enforce the BitNetConfig invariant that
/// Dimension % HeadCount == 0 and HeadDimension is even (for
/// rotary position embeddings).
/// </para>
/// </summary>
public static class TruckMateModelPresets
{
    /// <summary>Default vocab size from the word-level tokenizer.</summary>
    public const int DefaultVocabSize = 5174;

    /// <summary>
    /// Returns the named preset configuration. Recognized names
    /// (case-insensitive): "small", "medium", "large".
    /// </summary>
    public static ModelPreset GetPreset(string name, int? vocabSizeOverride = null)
    {
        var vocab = vocabSizeOverride ?? DefaultVocabSize;
        return (name?.ToLowerInvariant()) switch
        {
            "small"  => Small(vocab),
            "medium" => Medium(vocab),
            "large"  => Large(vocab),
            _        => Medium(vocab)
        };
    }

    /// <summary>
    /// ~7M parameters. Fast iteration, fits in 32MB RAM. Good for
    /// smoke-testing the training pipeline on low-end hardware.
    ///
    ///   dim=256, hidden=1024, layers=4, heads=8, seq=128
    /// </summary>
    public static ModelPreset Small(int vocabSize = DefaultVocabSize) =>
        new(
            Name: "truckmate-small",
            VocabSize: vocabSize,
            Dimension: 256,
            HiddenDimension: 1024,
            LayerCount: 4,
            HeadCount: 8,
            MaxSequenceLength: 128,
            EstimatedParameters: EstimateParams(vocabSize, 256, 1024, 4));

    /// <summary>
    /// ~56M parameters. Balanced quality + speed. The recommended
    /// starting point for Truck Mate intent classification.
    ///
    ///   dim=512, hidden=2048, layers=12, heads=16, seq=128
    /// </summary>
    public static ModelPreset Medium(int vocabSize = DefaultVocabSize) =>
        new(
            Name: "truckmate-medium",
            VocabSize: vocabSize,
            Dimension: 512,
            HiddenDimension: 2048,
            LayerCount: 12,
            HeadCount: 16,
            MaxSequenceLength: 128,
            EstimatedParameters: EstimateParams(vocabSize, 512, 2048, 12));

    /// <summary>
    /// ~121M parameters. Target scale from the original requirement.
    /// Requires more training data to avoid overfitting on the 50K
    /// corpus; use after scaling the corpus to 200K+ examples.
    ///
    ///   dim=768, hidden=3072, layers=12, heads=12, seq=256
    /// </summary>
    public static ModelPreset Large(int vocabSize = DefaultVocabSize) =>
        new(
            Name: "truckmate-large",
            VocabSize: vocabSize,
            Dimension: 768,
            HiddenDimension: 3072,
            LayerCount: 12,
            HeadCount: 12,
            MaxSequenceLength: 256,
            EstimatedParameters: EstimateParams(vocabSize, 768, 3072, 12));

    private static long EstimateParams(int vocab, int dim, int hidden, int layers)
    {
        long embeddings = (long)vocab * dim;              // token embeddings
        long perLayer = 4L * dim * dim                    // Q, K, V, O attention projections
                      + 3L * dim * hidden                 // SwiGLU gate + up + down
                      + 2L * dim;                         // 2 RmsNorm per layer
        long outputHead = (long)dim * vocab + dim;        // final norm + output projection
        return embeddings + layers * perLayer + outputHead;
    }
}

/// <summary>
/// Immutable description of a model architecture preset. Workers
/// and the coordinator both use this to construct matching
/// <c>BitNetConfig</c> objects so the weight vector dimension
/// agrees across the fleet.
/// </summary>
public sealed record ModelPreset(
    string Name,
    int VocabSize,
    int Dimension,
    int HiddenDimension,
    int LayerCount,
    int HeadCount,
    int MaxSequenceLength,
    long EstimatedParameters)
{
    /// <summary>
    /// Total number of fp32 elements in the global weight vector
    /// the coordinator tracks. For the D-4 flat-vector
    /// representation this is the sum of all parameter tensors.
    /// </summary>
    public long TotalWeightElements => EstimatedParameters;

    /// <summary>Human-readable summary for banners + dashboards.</summary>
    public string ToDisplayString() =>
        $"{Name}: vocab={VocabSize}, dim={Dimension}, hidden={HiddenDimension}, layers={LayerCount}, heads={HeadCount}, seq={MaxSequenceLength}, ~{EstimatedParameters / 1_000_000d:F1}M params";
}
