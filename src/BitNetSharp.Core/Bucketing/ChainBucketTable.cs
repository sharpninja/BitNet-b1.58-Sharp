namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// An immutable lookup table of up to 256 chain buckets.
/// Supports prefix-based lookup: given the last 1–3 tokens in the generation context,
/// the table returns the best-matching chain bucket for speculative decoding.
/// </summary>
public sealed class ChainBucketTable
{
    private readonly ChainBucket[] _buckets;

    // Prefix dictionaries keyed by the first 1, 2, or 3 token IDs of each chain.
    private readonly Dictionary<int, ChainBucket> _byPrefix1 = new();
    private readonly Dictionary<(int, int), ChainBucket> _byPrefix2 = new();
    private readonly Dictionary<(int, int, int), ChainBucket> _byPrefix3 = new();

    /// <summary>Maximum number of chain buckets (one byte = 256 values).</summary>
    public const int MaxBuckets = 256;

    public ChainBucketTable(IEnumerable<ChainBucket> buckets)
    {
        ArgumentNullException.ThrowIfNull(buckets);

        _buckets = buckets.Take(MaxBuckets).ToArray();

        foreach (var bucket in _buckets)
        {
            var ids = bucket.TokenIds;
            if (ids.Length < 2)
            {
                continue;
            }

            // Register the longest available prefix for the most specific match.
            if (ids.Length >= 3)
            {
                _byPrefix3.TryAdd((ids[0], ids[1], ids[2]), bucket);
            }

            _byPrefix2.TryAdd((ids[0], ids[1]), bucket);
            _byPrefix1.TryAdd(ids[0], bucket);
        }
    }

    /// <summary>Gets the number of chain buckets in the table.</summary>
    public int Count => _buckets.Length;

    /// <summary>Gets all chain buckets in the table.</summary>
    public IReadOnlyList<ChainBucket> Buckets => _buckets;

    /// <summary>
    /// Attempts to find a chain bucket whose start matches the tail of the provided context.
    /// The lookup tries a 3-token prefix first, then 2, then 1, and returns the first match.
    /// </summary>
    /// <param name="contextTail">
    /// The last up to 3 token IDs from the current generation context (most-recent last).
    /// </param>
    /// <param name="chain">The matching chain bucket if found; otherwise <c>null</c>.</param>
    /// <returns><c>true</c> if a matching chain was found; otherwise <c>false</c>.</returns>
    public bool TryLookupPrefix(IReadOnlyList<int> contextTail, out ChainBucket? chain)
    {
        ArgumentNullException.ThrowIfNull(contextTail);

        var count = contextTail.Count;
        if (count >= 3 && _byPrefix3.TryGetValue(
                (contextTail[count - 3], contextTail[count - 2], contextTail[count - 1]),
                out chain))
        {
            return true;
        }

        if (count >= 2 && _byPrefix2.TryGetValue(
                (contextTail[count - 2], contextTail[count - 1]),
                out chain))
        {
            return true;
        }

        if (count >= 1 && _byPrefix1.TryGetValue(contextTail[count - 1], out chain))
        {
            return true;
        }

        chain = null;
        return false;
    }

    /// <summary>Looks up a chain bucket by its compact byte identifier.</summary>
    public ChainBucket? GetById(byte chainId) =>
        Array.Find(_buckets, b => b.ChainId == chainId);

    /// <summary>
    /// Attempts to find the longest chain that exactly matches the token sequence starting at
    /// <paramref name="startIndex"/>. Uses the internal prefix index for O(1) candidate lookup,
    /// then verifies the full chain matches. Returns the best (longest) verified match.
    /// </summary>
    /// <param name="sequence">The token sequence to search within.</param>
    /// <param name="startIndex">The position in <paramref name="sequence"/> to start matching from.</param>
    /// <param name="chain">The best matching chain if found; otherwise <c>null</c>.</param>
    /// <returns><c>true</c> if a chain was found and fully verified; otherwise <c>false</c>.</returns>
    public bool TryMatchAt(IReadOnlyList<int> sequence, int startIndex, out ChainBucket? chain)
    {
        ArgumentNullException.ThrowIfNull(sequence);

        chain = null;
        var remaining = sequence.Count - startIndex;
        if (remaining < 2)
        {
            return false;
        }

        // Try prefix lookups from longest to shortest and verify the full chain each time.
        if (remaining >= 3 && _byPrefix3.TryGetValue(
                (sequence[startIndex], sequence[startIndex + 1], sequence[startIndex + 2]),
                out var candidate3)
            && IsFullMatch(sequence, startIndex, candidate3))
        {
            chain = candidate3;
            return true;
        }

        if (_byPrefix2.TryGetValue(
                (sequence[startIndex], sequence[startIndex + 1]),
                out var candidate2)
            && IsFullMatch(sequence, startIndex, candidate2))
        {
            chain = candidate2;
            return true;
        }

        if (_byPrefix1.TryGetValue(sequence[startIndex], out var candidate1)
            && IsFullMatch(sequence, startIndex, candidate1))
        {
            chain = candidate1;
            return true;
        }

        return false;
    }

    private static bool IsFullMatch(IReadOnlyList<int> sequence, int startIndex, ChainBucket candidate)
    {
        var chainLen = candidate.TokenIds.Length;
        if (startIndex + chainLen > sequence.Count)
        {
            return false;
        }

        for (var i = 0; i < chainLen; i++)
        {
            if (sequence[startIndex + i] != candidate.TokenIds[i])
            {
                return false;
            }
        }

        return true;
    }
}
