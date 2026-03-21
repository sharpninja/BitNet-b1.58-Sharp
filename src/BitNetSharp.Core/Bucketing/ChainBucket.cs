namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// Represents a single chain bucket: a frequent n-gram sequence associated with a compact byte identifier.
/// During inference the byte ChainId acts as a speculative-decoding shorthand for the full token sequence.
/// </summary>
/// <param name="ChainId">Compact byte identifier in the range 0–255.</param>
/// <param name="TokenIds">Ordered token IDs that make up the n-gram chain (length 2–8).</param>
/// <param name="Confidence">Normalised confidence score derived from corpus frequency and conditional probability.</param>
public sealed record ChainBucket(byte ChainId, int[] TokenIds, float Confidence)
{
    /// <summary>Gets the number of tokens in this chain.</summary>
    public int Length => TokenIds.Length;
}
