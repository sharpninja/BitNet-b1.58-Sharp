namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// Represents a single chain bucket: a frequent n-gram sequence associated with a compact byte identifier.
/// During inference the byte ChainId acts as a speculative-decoding shorthand for the full token sequence.
/// </summary>
public sealed record ChainBucket
{
    /// <summary>Compact byte identifier in the range 0–255.</summary>
    public byte ChainId { get; }

    /// <summary>Ordered token IDs that make up the n-gram chain (length 2–8).</summary>
    public int[] TokenIds { get; }

    /// <summary>Normalised confidence score derived from corpus frequency and conditional probability.</summary>
    public float Confidence { get; }

    /// <summary>Gets the number of tokens in this chain.</summary>
    public int Length => TokenIds.Length;

    public ChainBucket(byte chainId, int[] tokenIds, float confidence)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ChainId = chainId;
        TokenIds = (int[])tokenIds.Clone();
        Confidence = confidence;
    }
}
