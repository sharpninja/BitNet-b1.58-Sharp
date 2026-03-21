using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Tests;

public sealed class BucketMinerTests
{
    [Fact]
    public void Mine_EmptyInput_ReturnsEmptyTable()
    {
        var table = BucketMiner.Mine([]);
        Assert.Equal(0, table.Count);
    }

    [Fact]
    public void Mine_SingleShortSequence_ReturnsNoChains()
    {
        // A single-token sequence cannot form any n-gram of length >= 2.
        var table = BucketMiner.Mine([[1]]);
        Assert.Equal(0, table.Count);
    }

    [Fact]
    public void Mine_FrequentBigram_AppearsInTable()
    {
        // Repeat the same bigram [1, 2] many times so it scores highly.
        var sequence = Enumerable.Repeat(new int[] { 1, 2 }, 20)
            .SelectMany(s => s)
            .ToArray();

        var table = BucketMiner.Mine([sequence]);

        Assert.True(table.Count > 0);
        var found = table.Buckets.Any(b => b.TokenIds.Length >= 2 && b.TokenIds[0] == 1 && b.TokenIds[1] == 2);
        Assert.True(found, "The frequent bigram [1, 2] should appear as a chain bucket.");
    }

    [Fact]
    public void Mine_RespectsMaxBucketsLimit()
    {
        // Build a long sequence with many distinct bigrams.
        var sequence = Enumerable.Range(0, 200).ToArray();
        var table = BucketMiner.Mine([sequence], maxBuckets: 10);

        Assert.True(table.Count <= 10);
    }

    [Fact]
    public void Mine_ConfidenceValuesAreNormalised()
    {
        var sequence = new int[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
        var table = BucketMiner.Mine([sequence]);

        foreach (var bucket in table.Buckets)
        {
            Assert.True(bucket.Confidence is >= 0f and <= 1f,
                $"Confidence {bucket.Confidence} for chain {bucket.ChainId} is out of [0,1].");
        }
    }

    [Fact]
    public void Mine_ChainIdCountMatchesTableCount()
    {
        var sequence = new int[] { 5, 6, 7, 8, 5, 6, 7, 8 };
        var table = BucketMiner.Mine([sequence]);

        Assert.Equal(table.Count, table.Buckets.Count);
    }

    [Fact]
    public void ChainBucketTable_TryLookupPrefix_FindsSingleTokenPrefix()
    {
        var bucket = new ChainBucket(0, [10, 20, 30], 1f);
        var table = new ChainBucketTable([bucket]);

        var found = table.TryLookupPrefix([10], out var result);

        Assert.True(found);
        Assert.NotNull(result);
        Assert.Equal(bucket.ChainId, result!.ChainId);
    }

    [Fact]
    public void ChainBucketTable_TryLookupPrefix_FindsTwoTokenPrefix()
    {
        var bucket = new ChainBucket(0, [10, 20, 30], 1f);
        var table = new ChainBucketTable([bucket]);

        var found = table.TryLookupPrefix([5, 10, 20], out var result);

        Assert.True(found);
        Assert.NotNull(result);
    }

    [Fact]
    public void ChainBucketTable_TryLookupPrefix_FindsThreeTokenPrefix()
    {
        var bucket = new ChainBucket(0, [10, 20, 30, 40], 1f);
        var table = new ChainBucketTable([bucket]);

        var found = table.TryLookupPrefix([10, 20, 30], out var result);

        Assert.True(found);
        Assert.Equal(bucket.ChainId, result!.ChainId);
    }

    [Fact]
    public void ChainBucketTable_TryLookupPrefix_ReturnsFalseWhenNoMatch()
    {
        var bucket = new ChainBucket(0, [10, 20, 30], 1f);
        var table = new ChainBucketTable([bucket]);

        var found = table.TryLookupPrefix([99, 100], out var result);

        Assert.False(found);
        Assert.Null(result);
    }

    [Fact]
    public void ChainBucketTable_GetById_ReturnsCorrectBucket()
    {
        var bucket0 = new ChainBucket(0, [1, 2], 0.8f);
        var bucket1 = new ChainBucket(1, [3, 4], 0.6f);
        var table = new ChainBucketTable([bucket0, bucket1]);

        Assert.Equal(bucket0, table.GetById(0));
        Assert.Equal(bucket1, table.GetById(1));
        Assert.Null(table.GetById(42));
    }

    [Fact]
    public void ChainBucketTable_EnforcesMaxBucketsLimit()
    {
        var buckets = Enumerable.Range(0, 300)
            .Select(i => new ChainBucket((byte)(i % 256), [i, i + 1], 1f));
        var table = new ChainBucketTable(buckets);

        Assert.Equal(ChainBucketTable.MaxBuckets, table.Count);
    }

    [Fact]
    public void ChainBucket_LengthMatchesTokenIdsLength()
    {
        var bucket = new ChainBucket(5, [10, 20, 30, 40], 0.9f);
        Assert.Equal(4, bucket.Length);
    }

    [Fact]
    public void Mine_MultipleSequences_AggregatesNGrams()
    {
        // The bigram [7, 8] appears in both sequences.
        IReadOnlyList<int>[] sequences =
        [
            [1, 2, 7, 8, 3],
            [4, 5, 7, 8, 6]
        ];

        var table = BucketMiner.Mine(sequences);

        Assert.True(table.Count > 0);
        var found = table.Buckets.Any(b => b.TokenIds.Length >= 2 && b.TokenIds[0] == 7 && b.TokenIds[1] == 8);
        Assert.True(found, "The shared bigram [7, 8] should appear as a chain bucket.");
    }

    [Fact]
    public void ChainBucketTable_TryMatchAt_MatchesExactChainAtPosition()
    {
        var bucket = new ChainBucket(0, [10, 20, 30], 1f);
        var table = new ChainBucketTable([bucket]);

        // Sequence has the chain starting at index 2.
        IReadOnlyList<int> sequence = [1, 2, 10, 20, 30, 99];
        var found = table.TryMatchAt(sequence, 2, out var result);

        Assert.True(found);
        Assert.NotNull(result);
        Assert.Equal(0, result!.ChainId);
    }

    [Fact]
    public void ChainBucketTable_TryMatchAt_ReturnsFalseForPartialMatch()
    {
        // Chain is [10, 20, 30] but sequence only has [10, 20, 99] at that position.
        var bucket = new ChainBucket(0, [10, 20, 30], 1f);
        var table = new ChainBucketTable([bucket]);

        IReadOnlyList<int> sequence = [10, 20, 99];
        var found = table.TryMatchAt(sequence, 0, out var result);

        Assert.False(found);
        Assert.Null(result);
    }

    [Fact]
    public void ChainBucketTable_TryMatchAt_ReturnsFalseWhenChainExceedsSequenceLength()
    {
        var bucket = new ChainBucket(0, [10, 20, 30], 1f);
        var table = new ChainBucketTable([bucket]);

        // Only 1 token remaining at position 0, but chain needs 3.
        IReadOnlyList<int> sequence = [10];
        var found = table.TryMatchAt(sequence, 0, out var result);

        Assert.False(found);
        Assert.Null(result);
    }
}
