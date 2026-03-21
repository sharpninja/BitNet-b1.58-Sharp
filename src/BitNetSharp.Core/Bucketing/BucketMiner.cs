namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// Mines frequent n-gram chains from tokenized corpora and builds a <see cref="ChainBucketTable"/>.
/// Extracts n-grams of length 2–8, scores them by frequency × conditional probability,
/// and packs the top 256 candidates into a single bucket table.
/// </summary>
public static class BucketMiner
{
    /// <summary>Minimum n-gram length considered during mining.</summary>
    public const int MinNGramLength = 2;

    /// <summary>Maximum n-gram length considered during mining.</summary>
    public const int MaxNGramLength = 8;

    /// <summary>
    /// Scans the provided tokenized sequences, extracts frequent n-grams, and builds a
    /// <see cref="ChainBucketTable"/> containing up to 256 chain buckets.
    /// </summary>
    /// <param name="tokenizedSequences">
    /// An enumerable of tokenized sequences (each sequence is an ordered list of token IDs).
    /// </param>
    /// <param name="maxBuckets">
    /// Maximum number of chain buckets to include in the table (capped at 256).
    /// </param>
    /// <returns>A new <see cref="ChainBucketTable"/> populated with the top-scored chains.</returns>
    public static ChainBucketTable Mine(
        IEnumerable<IReadOnlyList<int>> tokenizedSequences,
        int maxBuckets = ChainBucketTable.MaxBuckets)
    {
        ArgumentNullException.ThrowIfNull(tokenizedSequences);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBuckets);

        maxBuckets = Math.Min(maxBuckets, ChainBucketTable.MaxBuckets);

        // Count raw n-gram frequencies for n = 2..MaxNGramLength.
        var ngramCounts = new Dictionary<NGramKey, int>(NGramKeyComparer.Instance);
        var prefixCounts = new Dictionary<NGramKey, int>(NGramKeyComparer.Instance);

        foreach (var sequence in tokenizedSequences)
        {
            var seqLen = sequence.Count;
            for (var start = 0; start < seqLen; start++)
            {
                for (var length = MinNGramLength; length <= MaxNGramLength && start + length <= seqLen; length++)
                {
                    var ngram = new NGramKey(sequence, start, length);
                    ngramCounts[ngram] = ngramCounts.TryGetValue(ngram, out var existing) ? existing + 1 : 1;

                    // Track the prefix (ngram[0..length-2]) for conditional-probability estimation.
                    if (length > 1)
                    {
                        var prefix = new NGramKey(sequence, start, length - 1);
                        prefixCounts[prefix] = prefixCounts.TryGetValue(prefix, out var pExisting) ? pExisting + 1 : 1;
                    }
                }
            }
        }

        if (ngramCounts.Count == 0)
        {
            return new ChainBucketTable([]);
        }

        // Score = frequency × conditional probability (frequency / prefix frequency).
        var scored = new List<(NGramKey Key, int Freq, double Score)>(ngramCounts.Count);
        foreach (var (key, freq) in ngramCounts)
        {
            double conditionalProb;
            if (key.Length > 1)
            {
                var prefix = new NGramKey(key.Tokens, key.Start, key.Length - 1);
                conditionalProb = prefixCounts.TryGetValue(prefix, out var prefixFreq) && prefixFreq > 0
                    ? freq / (double)prefixFreq
                    : 1d;
            }
            else
            {
                conditionalProb = 1d;
            }

            scored.Add((key, freq, freq * conditionalProb));
        }

        // Prefer longer chains at equal score for richer speculative decoding.
        scored.Sort(static (a, b) =>
        {
            var scoreCompare = b.Score.CompareTo(a.Score);
            return scoreCompare != 0 ? scoreCompare : b.Key.Length.CompareTo(a.Key.Length);
        });

        var maxScore = scored.Count > 0 ? scored[0].Score : 1d;
        if (maxScore <= 0d)
        {
            maxScore = 1d;
        }

        var selected = scored.Take(maxBuckets);
        var buckets = selected.Select((item, index) => new ChainBucket(
            (byte)(index & 0xFF),
            item.Key.ToArray(),
            (float)(item.Score / maxScore)));

        return new ChainBucketTable(buckets);
    }

    // Lightweight value-semantic wrapper around a slice of an existing IReadOnlyList<int>.
    // The struct retains a reference to the backing list rather than copying the slice, so callers
    // must ensure the backing list remains unchanged for the lifetime of any NGramKey instances
    // (i.e. throughout a single Mine() call, which is the only intended use).
    private readonly struct NGramKey(IReadOnlyList<int> tokens, int start, int length)
    {
        public readonly IReadOnlyList<int> Tokens = tokens;
        public readonly int Start = start;
        public readonly int Length = length;

        public int[] ToArray()
        {
            var array = new int[Length];
            for (var i = 0; i < Length; i++)
            {
                array[i] = Tokens[Start + i];
            }

            return array;
        }
    }

    private sealed class NGramKeyComparer : IEqualityComparer<NGramKey>
    {
        public static readonly NGramKeyComparer Instance = new();

        public bool Equals(NGramKey x, NGramKey y)
        {
            if (x.Length != y.Length)
            {
                return false;
            }

            for (var i = 0; i < x.Length; i++)
            {
                if (x.Tokens[x.Start + i] != y.Tokens[y.Start + i])
                {
                    return false;
                }
            }

            return true;
        }

        public int GetHashCode(NGramKey key)
        {
            var hash = new HashCode();
            for (var i = 0; i < key.Length; i++)
            {
                hash.Add(key.Tokens[key.Start + i]);
            }

            return hash.ToHashCode();
        }
    }
}
