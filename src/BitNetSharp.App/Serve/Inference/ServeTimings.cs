using System;

namespace BitNetSharp.App.Serve.Inference;

/// <summary>
/// Helper for minting Ollama-shaped timing metadata. Ollama reports durations
/// in nanoseconds and token counts as ints; clients parse both as required
/// fields on the terminal chunk, so we always populate them even when the
/// core inference path doesn't surface true per-phase timings.
/// </summary>
internal static class ServeTimings
{
    public static string UtcNow() =>
        DateTimeOffset.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", System.Globalization.CultureInfo.InvariantCulture);

    public static long ElapsedNanoseconds(DateTimeOffset start)
    {
        var elapsed = DateTimeOffset.UtcNow - start;
        return (long)(elapsed.TotalMilliseconds * 1_000_000d);
    }

    public static int EstimateTokens(string? text)
    {
        if (string.IsNullOrEmpty(text)) return 0;
        // Character-over-four is the common rule-of-thumb. Good enough for the
        // Ollama envelope's eval_count/prompt_eval_count hints that clients
        // display; exact counts would require routing through BitNetTokenizer.
        return Math.Max(1, text.Length / 4);
    }

    public static string NewChatCompletionId() => "chatcmpl-" + Guid.NewGuid().ToString("N")[..24];
}
