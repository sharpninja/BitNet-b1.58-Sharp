namespace BitNetSharp.Core.Sampling;

/// <summary>
/// Decoding-time logit adjustments shared by every token-selection path.
///
/// The paper-aligned model selects tokens via argmax, which collapses to
/// degenerate repetition when the highest-logit token is also the most recent.
/// Repetition-penalty rescaling matches the HuggingFace convention: logits of
/// tokens that already appear in a recent context window are divided by the
/// penalty when positive and multiplied when negative, so argmax tends to pick
/// something that is not a simple echo.
/// </summary>
public static class SamplingUtilities
{
    public static void ApplyRepetitionPenalty(
        Span<float> logits,
        ReadOnlySpan<int> recentContext,
        float penalty)
    {
        if (penalty <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(penalty), penalty, "Repetition penalty must be positive.");
        }

        if (penalty == 1f || recentContext.IsEmpty)
        {
            return;
        }

        for (var i = 0; i < recentContext.Length; i++)
        {
            var tokenId = recentContext[i];
            if ((uint)tokenId >= (uint)logits.Length)
            {
                continue;
            }

            var value = logits[tokenId];
            logits[tokenId] = value > 0f ? value / penalty : value * penalty;
        }
    }
}
