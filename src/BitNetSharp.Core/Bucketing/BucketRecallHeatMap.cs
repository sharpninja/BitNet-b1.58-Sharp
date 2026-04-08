namespace BitNetSharp.Core.Bucketing;

public sealed record HeatMapCounters(
    int VocabSize,
    long[] AttemptCounts,
    long[] AcceptCounts,
    long[] ChainAttemptCounts,
    long[] ChainAcceptCounts,
    long[,] ChainTransitions);

public sealed record TokenRecallEntry(int TokenId, long AttemptCount, long AcceptCount, double RecallRate);

public sealed record BucketRecallRanking(
    byte ChainId,
    double AggregateRecallRate,
    long TotalAcceptCount,
    long TotalAttemptCount,
    bool OnHotPath);

public sealed record ChainTransitionEntry(byte ChainId, long TransitionCount);

public sealed record ChainHotPath(byte[] ChainSequence, long MinTransitionCount);

public sealed class BucketRecallHeatMap
{
    private readonly long[] _attemptCounts;
    private readonly long[] _acceptCounts;
    private readonly long[] _chainAttemptCounts;
    private readonly long[] _chainAcceptCounts;
    private readonly long[,] _chainTransitions;
    private byte? _lastAcceptedChainId;

    public int VocabSize { get; }

    public BucketRecallHeatMap(int vocabSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        VocabSize = vocabSize;
        _attemptCounts = new long[vocabSize];
        _acceptCounts = new long[vocabSize];
        _chainAttemptCounts = new long[ChainBucketTable.MaxBuckets];
        _chainAcceptCounts = new long[ChainBucketTable.MaxBuckets];
        _chainTransitions = new long[ChainBucketTable.MaxBuckets, ChainBucketTable.MaxBuckets];
    }

    public void RecordChainAttempt(byte chainId, int[] tokenIds, int speculativeStartIndex)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        _chainAttemptCounts[chainId]++;
        for (var i = speculativeStartIndex; i < tokenIds.Length; i++)
        {
            var tokenId = tokenIds[i];
            if ((uint)tokenId < (uint)VocabSize)
            {
                _attemptCounts[tokenId]++;
            }
        }
    }

    public void RecordTokenAccepted(byte chainId, int tokenId)
    {
        if ((uint)tokenId < (uint)VocabSize)
        {
            _acceptCounts[tokenId]++;
        }
    }

    public void RecordChainAccepted(byte chainId)
    {
        _chainAcceptCounts[chainId]++;
        if (_lastAcceptedChainId.HasValue)
        {
            _chainTransitions[_lastAcceptedChainId.Value, chainId]++;
        }

        _lastAcceptedChainId = chainId;
    }

    public void ResetGenerationState()
    {
        _lastAcceptedChainId = null;
    }

    public long GetAttemptCount(int tokenId) =>
        (uint)tokenId < (uint)VocabSize ? _attemptCounts[tokenId] : 0L;

    public long GetAcceptCount(int tokenId) =>
        (uint)tokenId < (uint)VocabSize ? _acceptCounts[tokenId] : 0L;

    public double GetTokenRecallRate(int tokenId)
    {
        var attempts = GetAttemptCount(tokenId);
        return attempts == 0L ? 0d : (double)GetAcceptCount(tokenId) / attempts;
    }

    public long GetChainAttemptCount(byte chainId) => _chainAttemptCounts[chainId];

    public long GetChainAcceptCount(byte chainId) => _chainAcceptCounts[chainId];

    public double GetChainRecallRate(byte chainId)
    {
        var attempts = _chainAttemptCounts[chainId];
        return attempts == 0L ? 0d : (double)_chainAcceptCounts[chainId] / attempts;
    }

    public long GetTransitionCount(byte fromChainId, byte toChainId) =>
        _chainTransitions[fromChainId, toChainId];

    public IReadOnlyList<ChainTransitionEntry> GetIncomingChains(byte chainId, int maxResults = 10)
    {
        var entries = new List<ChainTransitionEntry>();
        for (var from = 0; from < ChainBucketTable.MaxBuckets; from++)
        {
            var count = _chainTransitions[from, chainId];
            if (count > 0)
            {
                entries.Add(new ChainTransitionEntry((byte)from, count));
            }
        }

        return entries
            .OrderByDescending(static e => e.TransitionCount)
            .Take(maxResults)
            .ToArray();
    }

    public IReadOnlyList<ChainTransitionEntry> GetOutgoingChains(byte chainId, int maxResults = 10)
    {
        var entries = new List<ChainTransitionEntry>();
        for (var to = 0; to < ChainBucketTable.MaxBuckets; to++)
        {
            var count = _chainTransitions[chainId, to];
            if (count > 0)
            {
                entries.Add(new ChainTransitionEntry((byte)to, count));
            }
        }

        return entries
            .OrderByDescending(static e => e.TransitionCount)
            .Take(maxResults)
            .ToArray();
    }

    public IReadOnlyList<ChainHotPath> GetHotPaths(int maxDepth = 5, int maxResults = 10, long minTransitions = 2)
    {
        var paths = new List<ChainHotPath>();
        var visited = new HashSet<byte>();

        for (var start = 0; start < ChainBucketTable.MaxBuckets; start++)
        {
            if (_chainAcceptCounts[start] == 0)
            {
                continue;
            }

            visited.Clear();
            var sequence = new List<byte> { (byte)start };
            visited.Add((byte)start);
            var bottleneck = long.MaxValue;

            var current = (byte)start;
            for (var depth = 1; depth < maxDepth; depth++)
            {
                var bestNext = -1;
                var bestCount = 0L;
                for (var next = 0; next < ChainBucketTable.MaxBuckets; next++)
                {
                    var count = _chainTransitions[current, next];
                    if (count >= minTransitions && count > bestCount && !visited.Contains((byte)next))
                    {
                        bestNext = next;
                        bestCount = count;
                    }
                }

                if (bestNext < 0)
                {
                    break;
                }

                sequence.Add((byte)bestNext);
                visited.Add((byte)bestNext);
                bottleneck = Math.Min(bottleneck, bestCount);
                current = (byte)bestNext;
            }

            if (sequence.Count >= 2)
            {
                paths.Add(new ChainHotPath(sequence.ToArray(), bottleneck));
            }
        }

        // Remove sub-paths: if path A is a prefix of path B, keep only B.
        var deduplicated = new List<ChainHotPath>();
        var sortedByLength = paths.OrderByDescending(static p => p.ChainSequence.Length).ToArray();
        foreach (var path in sortedByLength)
        {
            var isSubPath = false;
            foreach (var longer in deduplicated)
            {
                if (IsSubsequence(path.ChainSequence, longer.ChainSequence))
                {
                    isSubPath = true;
                    break;
                }
            }

            if (!isSubPath)
            {
                deduplicated.Add(path);
            }
        }

        return deduplicated
            .OrderByDescending(static p => p.MinTransitionCount)
            .Take(maxResults)
            .ToArray();
    }

    public IReadOnlyList<TokenRecallEntry> GetTopTokensByAcceptCount(int maxResults = 20)
    {
        var entries = new List<TokenRecallEntry>();
        for (var i = 0; i < VocabSize; i++)
        {
            if (_acceptCounts[i] > 0)
            {
                entries.Add(new TokenRecallEntry(i, _attemptCounts[i], _acceptCounts[i], GetTokenRecallRate(i)));
            }
        }

        return entries
            .OrderByDescending(static e => e.AcceptCount)
            .Take(maxResults)
            .ToArray();
    }

    public IReadOnlyList<TokenRecallEntry> GetTopTokensByRecallRate(int maxResults = 20, int minAttempts = 1)
    {
        var entries = new List<TokenRecallEntry>();
        for (var i = 0; i < VocabSize; i++)
        {
            if (_attemptCounts[i] >= minAttempts)
            {
                entries.Add(new TokenRecallEntry(i, _attemptCounts[i], _acceptCounts[i], GetTokenRecallRate(i)));
            }
        }

        return entries
            .OrderByDescending(static e => e.RecallRate)
            .Take(maxResults)
            .ToArray();
    }

    public IReadOnlyList<BucketRecallRanking> RankBucketsForCompaction(ChainBucketTable table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var hotPathChainIds = GetHotPathChainIds();
        var rankings = new List<BucketRecallRanking>();

        foreach (var bucket in table.Buckets)
        {
            var attempts = _chainAttemptCounts[bucket.ChainId];
            var accepts = _chainAcceptCounts[bucket.ChainId];
            var recallRate = attempts == 0L ? 0d : (double)accepts / attempts;
            var onHotPath = hotPathChainIds.Contains(bucket.ChainId);

            rankings.Add(new BucketRecallRanking(
                bucket.ChainId,
                recallRate,
                accepts,
                attempts,
                onHotPath));
        }

        // Sort worst-first: non-hot-path chains with low recall come first (pruning candidates).
        return rankings
            .OrderBy(static r => r.OnHotPath)
            .ThenBy(static r => r.AggregateRecallRate)
            .ThenBy(static r => r.TotalAcceptCount)
            .ToArray();
    }

    public IReadOnlySet<byte> IdentifyLowValueBuckets(ChainBucketTable table, double threshold = 0.5, int minAttempts = 2)
    {
        ArgumentNullException.ThrowIfNull(table);

        var hotPathChainIds = GetHotPathChainIds();
        var lowValue = new HashSet<byte>();

        foreach (var bucket in table.Buckets)
        {
            if (hotPathChainIds.Contains(bucket.ChainId))
            {
                continue;
            }

            var attempts = _chainAttemptCounts[bucket.ChainId];
            if (attempts < minAttempts)
            {
                lowValue.Add(bucket.ChainId);
                continue;
            }

            var recallRate = (double)_chainAcceptCounts[bucket.ChainId] / attempts;
            if (recallRate < threshold)
            {
                lowValue.Add(bucket.ChainId);
            }
        }

        return lowValue;
    }

    public void Reset()
    {
        Array.Clear(_attemptCounts);
        Array.Clear(_acceptCounts);
        Array.Clear(_chainAttemptCounts);
        Array.Clear(_chainAcceptCounts);
        Array.Clear(_chainTransitions);
        _lastAcceptedChainId = null;
    }

    public void MergeFrom(BucketRecallHeatMap other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.VocabSize != VocabSize)
        {
            throw new ArgumentException(
                $"Cannot merge heat maps with different vocab sizes ({VocabSize} vs {other.VocabSize}).",
                nameof(other));
        }

        for (var i = 0; i < VocabSize; i++)
        {
            _attemptCounts[i] += other._attemptCounts[i];
            _acceptCounts[i] += other._acceptCounts[i];
        }

        for (var i = 0; i < ChainBucketTable.MaxBuckets; i++)
        {
            _chainAttemptCounts[i] += other._chainAttemptCounts[i];
            _chainAcceptCounts[i] += other._chainAcceptCounts[i];
            for (var j = 0; j < ChainBucketTable.MaxBuckets; j++)
            {
                _chainTransitions[i, j] += other._chainTransitions[i, j];
            }
        }
    }

    public HeatMapCounters ExportCounters()
    {
        var attemptCounts = (long[])_attemptCounts.Clone();
        var acceptCounts = (long[])_acceptCounts.Clone();
        var chainAttemptCounts = (long[])_chainAttemptCounts.Clone();
        var chainAcceptCounts = (long[])_chainAcceptCounts.Clone();
        var chainTransitions = (long[,])_chainTransitions.Clone();
        return new HeatMapCounters(VocabSize, attemptCounts, acceptCounts, chainAttemptCounts, chainAcceptCounts, chainTransitions);
    }

    public static BucketRecallHeatMap FromCounters(HeatMapCounters counters)
    {
        ArgumentNullException.ThrowIfNull(counters);

        if (counters.AttemptCounts.Length != counters.VocabSize
            || counters.AcceptCounts.Length != counters.VocabSize)
        {
            throw new ArgumentException("Token counter arrays must match VocabSize.", nameof(counters));
        }

        if (counters.ChainAttemptCounts.Length != ChainBucketTable.MaxBuckets
            || counters.ChainAcceptCounts.Length != ChainBucketTable.MaxBuckets)
        {
            throw new ArgumentException($"Chain counter arrays must have length {ChainBucketTable.MaxBuckets}.", nameof(counters));
        }

        if (counters.ChainTransitions.GetLength(0) != ChainBucketTable.MaxBuckets
            || counters.ChainTransitions.GetLength(1) != ChainBucketTable.MaxBuckets)
        {
            throw new ArgumentException($"Chain transitions matrix must be {ChainBucketTable.MaxBuckets}x{ChainBucketTable.MaxBuckets}.", nameof(counters));
        }

        var heatMap = new BucketRecallHeatMap(counters.VocabSize);
        Array.Copy(counters.AttemptCounts, heatMap._attemptCounts, counters.VocabSize);
        Array.Copy(counters.AcceptCounts, heatMap._acceptCounts, counters.VocabSize);
        Array.Copy(counters.ChainAttemptCounts, heatMap._chainAttemptCounts, ChainBucketTable.MaxBuckets);
        Array.Copy(counters.ChainAcceptCounts, heatMap._chainAcceptCounts, ChainBucketTable.MaxBuckets);
        Array.Copy(counters.ChainTransitions, heatMap._chainTransitions, ChainBucketTable.MaxBuckets * ChainBucketTable.MaxBuckets);
        return heatMap;
    }

    private HashSet<byte> GetHotPathChainIds()
    {
        var hotPaths = GetHotPaths();
        var chainIds = new HashSet<byte>();
        foreach (var path in hotPaths)
        {
            foreach (var chainId in path.ChainSequence)
            {
                chainIds.Add(chainId);
            }
        }

        return chainIds;
    }

    private static bool IsSubsequence(byte[] candidate, byte[] longer)
    {
        if (candidate.Length >= longer.Length)
        {
            return false;
        }

        for (var start = 0; start <= longer.Length - candidate.Length; start++)
        {
            var match = true;
            for (var i = 0; i < candidate.Length; i++)
            {
                if (longer[start + i] != candidate[i])
                {
                    match = false;
                    break;
                }
            }

            if (match)
            {
                return true;
            }
        }

        return false;
    }
}
