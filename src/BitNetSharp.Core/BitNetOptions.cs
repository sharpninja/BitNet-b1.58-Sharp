using BitNetSharp.Core.Inference;

namespace BitNetSharp.Core;

public sealed record BitNetOptions(
    IReadOnlyList<string> Vocabulary,
    VerbosityLevel Verbosity = VerbosityLevel.Normal,
    int MaxResponseTokens = 24,
    string PrimaryLanguage = "en-US",
    bool EnableChainBuckets = false,
    bool EnableSequenceCompression = false,
    double ChainBucketAcceptanceThreshold = 0.85d,
    bool EnableRecallHeatMap = true,
    float RepetitionPenalty = 1.3f,
    int RepetitionPenaltyWindow = 64,
    bool UseIntegerForward = false)
{
    /// <summary>
    /// Name of the env var that forces <see cref="UseIntegerForward"/> true
    /// at construction sites that consult <see cref="IntegerForwardEnvDefault"/>.
    /// Set <c>BITNETSHARP_USE_INTEGER_FORWARD=1</c> before launching the
    /// serve to flip the hot path to the integer composer without rebaking
    /// config or GGUF metadata.
    /// </summary>
    public const string UseIntegerForwardEnvVar = "BITNETSHARP_USE_INTEGER_FORWARD";

    /// <summary>
    /// True when the process was launched with <see cref="UseIntegerForwardEnvVar"/>
    /// set to "1". Evaluated once per call; cheap enough for construction
    /// sites.
    /// </summary>
    public static bool IntegerForwardEnvDefault => string.Equals(
        Environment.GetEnvironmentVariable(UseIntegerForwardEnvVar),
        "1",
        StringComparison.Ordinal);

    /// <summary>
    /// Section B - KV5b: env var that overrides the model config's
    /// <see cref="BitNetConfig.KvCacheQuantization"/> at runtime. Accepts
    /// "Fp32" or "Int8" (case-insensitive). When unset or unrecognised,
    /// the config-declared value (default Fp32) is used. Set
    /// <c>BITNETSHARP_KV_CACHE_QUANTIZATION=Int8</c> before launching the
    /// serve to switch a Bonsai-loaded model to int8 KV without rebaking
    /// config or GGUF metadata.
    /// </summary>
    public const string KvCacheQuantizationEnvVar = "BITNETSHARP_KV_CACHE_QUANTIZATION";

    /// <summary>
    /// Reads <see cref="KvCacheQuantizationEnvVar"/> and returns a parsed
    /// <see cref="KvCacheQuantization"/>, or <c>null</c> if unset or
    /// unparseable. Construction sites should prefer the parsed override
    /// to the config-declared value when present.
    /// </summary>
    public static KvCacheQuantization? KvCacheQuantizationEnvOverride
    {
        get
        {
            var raw = Environment.GetEnvironmentVariable(KvCacheQuantizationEnvVar);
            if (string.IsNullOrWhiteSpace(raw))
            {
                return null;
            }
            return Enum.TryParse<KvCacheQuantization>(raw, ignoreCase: true, out var parsed)
                ? parsed
                : null;
        }
    }
}
