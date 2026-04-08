using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Tests;

public sealed class BucketRecallHeatMapTests
{
    private const int TestVocabSize = 32;

    [Fact]
    public void RecordChainAttempt_IncrementsAttemptCountsForSpeculativeTokens()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var tokenIds = new[] { 5, 10, 15, 20 };

        heatMap.RecordChainAttempt(0, tokenIds, speculativeStartIndex: 2);

        Assert.Equal(0, heatMap.GetAttemptCount(5));
        Assert.Equal(0, heatMap.GetAttemptCount(10));
        Assert.Equal(1, heatMap.GetAttemptCount(15));
        Assert.Equal(1, heatMap.GetAttemptCount(20));
    }

    [Fact]
    public void RecordChainAttempt_IncrementsChainAttemptCount()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAttempt(3, [1, 2, 3], speculativeStartIndex: 1);
        heatMap.RecordChainAttempt(3, [1, 2, 3], speculativeStartIndex: 1);

        Assert.Equal(2, heatMap.GetChainAttemptCount(3));
    }

    [Fact]
    public void RecordTokenAccepted_IncrementsAcceptCounters()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordTokenAccepted(0, 7);
        heatMap.RecordTokenAccepted(0, 7);
        heatMap.RecordTokenAccepted(0, 7);

        Assert.Equal(3, heatMap.GetAcceptCount(7));
    }

    [Fact]
    public void RecordChainAccepted_RecordsTransitionFromPreviousChain()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAccepted(1);
        heatMap.RecordChainAccepted(5);

        Assert.Equal(1, heatMap.GetTransitionCount(1, 5));
    }

    [Fact]
    public void RecordChainAccepted_NoTransitionOnFirstChainInGeneration()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAccepted(3);

        Assert.Equal(1, heatMap.GetChainAcceptCount(3));
        // No transition should exist since there was no previous chain.
        for (var from = 0; from < ChainBucketTable.MaxBuckets; from++)
        {
            Assert.Equal(0, heatMap.GetTransitionCount((byte)from, 3));
        }
    }

    [Fact]
    public void ResetGenerationState_ClearsLastAcceptedChainId()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAccepted(1);
        heatMap.ResetGenerationState();
        heatMap.RecordChainAccepted(5);

        // No transition from 1→5 because generation state was reset.
        Assert.Equal(0, heatMap.GetTransitionCount(1, 5));
    }

    [Fact]
    public void GetTokenRecallRate_ReturnsZeroWhenNeverAttempted()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        Assert.Equal(0d, heatMap.GetTokenRecallRate(10));
    }

    [Fact]
    public void GetTokenRecallRate_ReturnsCorrectRatio()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAttempt(0, [5], speculativeStartIndex: 0);
        heatMap.RecordChainAttempt(0, [5], speculativeStartIndex: 0);
        heatMap.RecordChainAttempt(0, [5], speculativeStartIndex: 0);
        heatMap.RecordChainAttempt(0, [5], speculativeStartIndex: 0);
        heatMap.RecordTokenAccepted(0, 5);
        heatMap.RecordTokenAccepted(0, 5);
        heatMap.RecordTokenAccepted(0, 5);

        Assert.Equal(0.75d, heatMap.GetTokenRecallRate(5));
    }

    [Fact]
    public void GetTransitionCount_ReturnsRecordedCount()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAccepted(2);
        heatMap.RecordChainAccepted(7);
        heatMap.ResetGenerationState();
        heatMap.RecordChainAccepted(2);
        heatMap.RecordChainAccepted(7);

        Assert.Equal(2, heatMap.GetTransitionCount(2, 7));
    }

    [Fact]
    public void GetIncomingChains_ReturnsPredecessorsSortedByCount()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        // Chain 10 is reached from chain 1 (3 times) and chain 2 (1 time).
        for (var i = 0; i < 3; i++)
        {
            heatMap.RecordChainAccepted(1);
            heatMap.RecordChainAccepted(10);
            heatMap.ResetGenerationState();
        }

        heatMap.RecordChainAccepted(2);
        heatMap.RecordChainAccepted(10);

        var incoming = heatMap.GetIncomingChains(10);
        Assert.Equal(2, incoming.Count);
        Assert.Equal(1, incoming[0].ChainId);
        Assert.Equal(3, incoming[0].TransitionCount);
        Assert.Equal(2, incoming[1].ChainId);
        Assert.Equal(1, incoming[1].TransitionCount);
    }

    [Fact]
    public void GetOutgoingChains_ReturnsSuccessorsSortedByCount()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        // Chain 5 leads to chain 8 (2 times) and chain 9 (1 time).
        heatMap.RecordChainAccepted(5);
        heatMap.RecordChainAccepted(8);
        heatMap.ResetGenerationState();
        heatMap.RecordChainAccepted(5);
        heatMap.RecordChainAccepted(8);
        heatMap.ResetGenerationState();
        heatMap.RecordChainAccepted(5);
        heatMap.RecordChainAccepted(9);

        var outgoing = heatMap.GetOutgoingChains(5);
        Assert.Equal(2, outgoing.Count);
        Assert.Equal(8, outgoing[0].ChainId);
        Assert.Equal(2, outgoing[0].TransitionCount);
        Assert.Equal(9, outgoing[1].ChainId);
        Assert.Equal(1, outgoing[1].TransitionCount);
    }

    [Fact]
    public void GetHotPaths_FindsFrequentChainSequences()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        // Create a clear hot-path: 1 → 2 → 3 (each transition fires 5 times).
        for (var i = 0; i < 5; i++)
        {
            heatMap.RecordChainAccepted(1);
            heatMap.RecordChainAccepted(2);
            heatMap.RecordChainAccepted(3);
            heatMap.ResetGenerationState();
        }

        var hotPaths = heatMap.GetHotPaths(maxDepth: 5, maxResults: 10, minTransitions: 2);
        Assert.True(hotPaths.Count > 0);

        var mainPath = hotPaths[0];
        Assert.True(mainPath.ChainSequence.Length >= 3);
        Assert.Equal(1, mainPath.ChainSequence[0]);
        Assert.Equal(2, mainPath.ChainSequence[1]);
        Assert.Equal(3, mainPath.ChainSequence[2]);
        Assert.Equal(5, mainPath.MinTransitionCount);
    }

    [Fact]
    public void GetHotPaths_RespectsMinTransitionsThreshold()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        // Only 1 transition — below the default minTransitions=2.
        heatMap.RecordChainAccepted(1);
        heatMap.RecordChainAccepted(2);

        var hotPaths = heatMap.GetHotPaths(minTransitions: 2);
        Assert.Empty(hotPaths);
    }

    [Fact]
    public void GetTopTokensByAcceptCount_ReturnsDescendingOrder()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordTokenAccepted(0, 3);
        heatMap.RecordTokenAccepted(0, 7);
        heatMap.RecordTokenAccepted(0, 7);
        heatMap.RecordTokenAccepted(0, 7);
        heatMap.RecordTokenAccepted(0, 5);
        heatMap.RecordTokenAccepted(0, 5);

        var top = heatMap.GetTopTokensByAcceptCount(maxResults: 3);
        Assert.Equal(3, top.Count);
        Assert.Equal(7, top[0].TokenId);
        Assert.Equal(3, top[0].AcceptCount);
        Assert.Equal(5, top[1].TokenId);
        Assert.Equal(2, top[1].AcceptCount);
        Assert.Equal(3, top[2].TokenId);
        Assert.Equal(1, top[2].AcceptCount);
    }

    [Fact]
    public void GetTopTokensByRecallRate_RespectsMinAttempts()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        // Token 3: 1 attempt, 1 accept (100% rate but below minAttempts=3).
        heatMap.RecordChainAttempt(0, [3], 0);
        heatMap.RecordTokenAccepted(0, 3);

        // Token 7: 5 attempts, 4 accepts (80% rate, above minAttempts).
        for (var i = 0; i < 5; i++)
        {
            heatMap.RecordChainAttempt(0, [7], 0);
        }

        for (var i = 0; i < 4; i++)
        {
            heatMap.RecordTokenAccepted(0, 7);
        }

        var top = heatMap.GetTopTokensByRecallRate(maxResults: 10, minAttempts: 3);
        Assert.Single(top);
        Assert.Equal(7, top[0].TokenId);
    }

    [Fact]
    public void RankBucketsForCompaction_ReturnsWorstFirst()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f)
        ]);

        // Chain 0: 10 attempts, 8 accepts (80%).
        for (var i = 0; i < 10; i++)
        {
            heatMap.RecordChainAttempt(0, [1, 2], 1);
        }

        for (var i = 0; i < 8; i++)
        {
            heatMap.RecordChainAccepted(0);
        }

        // Chain 1: 10 attempts, 2 accepts (20%).
        for (var i = 0; i < 10; i++)
        {
            heatMap.RecordChainAttempt(1, [3, 4], 1);
        }

        for (var i = 0; i < 2; i++)
        {
            heatMap.RecordChainAccepted(1);
        }

        var rankings = heatMap.RankBucketsForCompaction(table);
        Assert.Equal(2, rankings.Count);
        Assert.Equal(1, rankings[0].ChainId); // Worst first (20%).
        Assert.Equal(0, rankings[1].ChainId); // Better (80%).
    }

    [Fact]
    public void RankBucketsForCompaction_MarksHotPathChains()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f),
            new ChainBucket(2, [5, 6], 1.0f)
        ]);

        // Create a hot-path: 0 → 1 (5 transitions).
        for (var i = 0; i < 5; i++)
        {
            heatMap.RecordChainAccepted(0);
            heatMap.RecordChainAccepted(1);
            heatMap.ResetGenerationState();
        }

        var rankings = heatMap.RankBucketsForCompaction(table);
        var chain0 = rankings.Single(r => r.ChainId == 0);
        var chain1 = rankings.Single(r => r.ChainId == 1);
        var chain2 = rankings.Single(r => r.ChainId == 2);

        Assert.True(chain0.OnHotPath);
        Assert.True(chain1.OnHotPath);
        Assert.False(chain2.OnHotPath);
    }

    [Fact]
    public void IdentifyLowValueBuckets_ReturnsChainsBelowThreshold()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f)
        ]);

        // Chain 0: 10 attempts, 1 accept (10% — below 50% threshold).
        for (var i = 0; i < 10; i++)
        {
            heatMap.RecordChainAttempt(0, [1, 2], 1);
        }

        heatMap.RecordChainAccepted(0);

        // Chain 1: 10 attempts, 8 accepts (80% — above threshold).
        for (var i = 0; i < 10; i++)
        {
            heatMap.RecordChainAttempt(1, [3, 4], 1);
        }

        for (var i = 0; i < 8; i++)
        {
            heatMap.RecordChainAccepted(1);
        }

        var lowValue = heatMap.IdentifyLowValueBuckets(table, threshold: 0.5);
        Assert.Contains((byte)0, lowValue);
        Assert.DoesNotContain((byte)1, lowValue);
    }

    [Fact]
    public void IdentifyLowValueBuckets_ExcludesHotPathChains()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f)
        ]);

        // Chain 0: low recall but on a hot-path.
        for (var i = 0; i < 10; i++)
        {
            heatMap.RecordChainAttempt(0, [1, 2], 1);
        }

        heatMap.RecordChainAccepted(0);

        // Create hot-path: 0 → 1 (3 transitions).
        for (var i = 0; i < 3; i++)
        {
            heatMap.RecordChainAccepted(0);
            heatMap.RecordChainAccepted(1);
            heatMap.ResetGenerationState();
        }

        var lowValue = heatMap.IdentifyLowValueBuckets(table, threshold: 0.5, minAttempts: 2);
        Assert.DoesNotContain((byte)0, lowValue); // Protected by hot-path.
    }

    [Fact]
    public void Reset_ClearsAllCountersAndTransitions()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        heatMap.RecordChainAttempt(0, [5, 10], 0);
        heatMap.RecordTokenAccepted(0, 5);
        heatMap.RecordChainAccepted(0);
        heatMap.RecordChainAccepted(1);

        heatMap.Reset();

        Assert.Equal(0, heatMap.GetAttemptCount(5));
        Assert.Equal(0, heatMap.GetAcceptCount(5));
        Assert.Equal(0, heatMap.GetChainAttemptCount(0));
        Assert.Equal(0, heatMap.GetChainAcceptCount(0));
        Assert.Equal(0, heatMap.GetTransitionCount(0, 1));
    }

    [Fact]
    public void MergeFrom_AddsCounts()
    {
        var a = new BucketRecallHeatMap(TestVocabSize);
        var b = new BucketRecallHeatMap(TestVocabSize);

        a.RecordChainAttempt(0, [5], 0);
        a.RecordTokenAccepted(0, 5);
        b.RecordChainAttempt(0, [5], 0);
        b.RecordTokenAccepted(0, 5);

        a.MergeFrom(b);

        Assert.Equal(2, a.GetAttemptCount(5));
        Assert.Equal(2, a.GetAcceptCount(5));
    }

    [Fact]
    public void MergeFrom_AddsTransitionCounts()
    {
        var a = new BucketRecallHeatMap(TestVocabSize);
        var b = new BucketRecallHeatMap(TestVocabSize);

        a.RecordChainAccepted(1);
        a.RecordChainAccepted(2);

        b.RecordChainAccepted(1);
        b.RecordChainAccepted(2);
        b.ResetGenerationState();
        b.RecordChainAccepted(1);
        b.RecordChainAccepted(2);

        a.MergeFrom(b);

        Assert.Equal(3, a.GetTransitionCount(1, 2));
    }

    [Fact]
    public void MergeFrom_ThrowsOnVocabSizeMismatch()
    {
        var a = new BucketRecallHeatMap(32);
        var b = new BucketRecallHeatMap(64);

        Assert.Throws<ArgumentException>(() => a.MergeFrom(b));
    }

    [Fact]
    public void ExportCounters_RoundTripsViaFromCounters()
    {
        var original = new BucketRecallHeatMap(TestVocabSize);

        original.RecordChainAttempt(0, [5, 10], 0);
        original.RecordTokenAccepted(0, 5);
        original.RecordChainAccepted(0);
        original.RecordChainAccepted(1);

        var counters = original.ExportCounters();
        var restored = BucketRecallHeatMap.FromCounters(counters);

        Assert.Equal(original.VocabSize, restored.VocabSize);
        Assert.Equal(original.GetAttemptCount(5), restored.GetAttemptCount(5));
        Assert.Equal(original.GetAttemptCount(10), restored.GetAttemptCount(10));
        Assert.Equal(original.GetAcceptCount(5), restored.GetAcceptCount(5));
        Assert.Equal(original.GetChainAttemptCount(0), restored.GetChainAttemptCount(0));
        Assert.Equal(original.GetChainAcceptCount(0), restored.GetChainAcceptCount(0));
        Assert.Equal(original.GetChainAcceptCount(1), restored.GetChainAcceptCount(1));
        Assert.Equal(original.GetTransitionCount(0, 1), restored.GetTransitionCount(0, 1));
    }
}
