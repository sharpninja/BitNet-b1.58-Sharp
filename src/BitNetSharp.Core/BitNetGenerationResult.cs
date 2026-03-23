namespace BitNetSharp.Core;

public sealed record ChainBucketGenerationMetrics(
    int AttemptedChains,
    int AcceptedChains,
    int AttemptedTokens,
    int AcceptedTokens,
    double AcceptedTokenRate,
    double AverageAcceptedTokensPerAcceptedChain,
    double AcceptanceThreshold);

public sealed record BitNetGenerationResult(
    string ResponseText,
    IReadOnlyList<string> Tokens,
    IReadOnlyList<string> Diagnostics,
    ChainBucketGenerationMetrics? ChainBucketMetrics = null);
