using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Core.Quantization;

/// <summary>
/// Builds a physical row permutation for token-ID-addressed weight matrices
/// (output head, token embeddings) based on heat map hot-path data.
/// Tokens that frequently co-occur in speculative decoding chains are placed
/// into adjacent physical rows for cache locality during verification bursts.
/// </summary>
public static class HeatMapRowPermutation
{
    /// <summary>
    /// Build a row permutation that clusters hot-path chain tokens into adjacent
    /// physical positions. Returns mapping where permutation[logicalRow] = physicalRow.
    /// </summary>
    public static int[] BuildFromHeatMap(
        BucketRecallHeatMap heatMap,
        ChainBucketTable bucketTable,
        int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentNullException.ThrowIfNull(bucketTable);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        var permutation = new int[vocabSize];
        var assigned = new HashSet<int>();
        var nextPhysical = 0;

        // Extract hot paths ordered by transition count (highest traffic first)
        var hotPaths = heatMap.GetHotPaths();

        // Build chain ID -> token IDs lookup
        var chainTokens = new Dictionary<byte, int[]>();
        foreach (var bucket in bucketTable.Buckets)
        {
            chainTokens[bucket.ChainId] = bucket.TokenIds;
        }

        // Assign adjacent physical positions to tokens within each hot path
        foreach (var path in hotPaths.OrderByDescending(p => p.MinTransitionCount))
        {
            foreach (var chainId in path.ChainSequence)
            {
                if (!chainTokens.TryGetValue(chainId, out var tokenIds))
                {
                    continue;
                }

                foreach (var tokenId in tokenIds)
                {
                    if (tokenId >= 0 && tokenId < vocabSize && assigned.Add(tokenId))
                    {
                        permutation[tokenId] = nextPhysical++;
                    }
                }
            }
        }

        // Fill remaining positions with unassigned tokens
        for (var id = 0; id < vocabSize; id++)
        {
            if (assigned.Add(id))
            {
                permutation[id] = nextPhysical++;
            }
        }

        return permutation;
    }

    /// <summary>
    /// Verify that a permutation is a valid bijection (every physical position 0..N-1 appears exactly once).
    /// </summary>
    public static bool IsValidPermutation(int[] permutation, int size)
    {
        if (permutation.Length != size)
        {
            return false;
        }

        var seen = new bool[size];
        foreach (var physical in permutation)
        {
            if (physical < 0 || physical >= size || seen[physical])
            {
                return false;
            }

            seen[physical] = true;
        }

        return true;
    }
}
