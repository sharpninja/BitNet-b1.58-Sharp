using BitNetSharp.Core.Bucketing;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

public sealed class HeatMapRowPermutationTests
{
    [Fact]
    public void BuildPermutation_IsValidPermutation()
    {
        var (heatMap, bucketTable) = CreateTestHeatMapAndTable();
        var vocabSize = 32;

        var permutation = HeatMapRowPermutation.BuildFromHeatMap(heatMap, bucketTable, vocabSize);

        Assert.Equal(vocabSize, permutation.Length);
        Assert.True(HeatMapRowPermutation.IsValidPermutation(permutation, vocabSize));
    }

    [Fact]
    public void BuildPermutation_HotPathTokensGetAdjacentPositions()
    {
        var (heatMap, bucketTable) = CreateTestHeatMapAndTable();
        var vocabSize = 32;

        var permutation = HeatMapRowPermutation.BuildFromHeatMap(heatMap, bucketTable, vocabSize);

        // Chain 0 has tokens [5, 10, 15], Chain 1 has tokens [3, 7, 11]
        // Both are in the hot path. Tokens within each chain should be contiguous.
        var chain0Positions = new[] { permutation[5], permutation[10], permutation[15] };
        Array.Sort(chain0Positions);

        // The three tokens should be in consecutive physical positions
        Assert.Equal(chain0Positions[0] + 1, chain0Positions[1]);
        Assert.Equal(chain0Positions[1] + 1, chain0Positions[2]);
    }

    [Fact]
    public void BuildPermutation_ColdTokensFillRemaining()
    {
        var (heatMap, bucketTable) = CreateTestHeatMapAndTable();
        var vocabSize = 32;

        var permutation = HeatMapRowPermutation.BuildFromHeatMap(heatMap, bucketTable, vocabSize);

        // Hot tokens: 3, 5, 7, 10, 11, 15 (from chains 0 and 1)
        var hotTokens = new HashSet<int> { 3, 5, 7, 10, 11, 15 };

        // Cold tokens should still have valid physical positions
        for (var id = 0; id < vocabSize; id++)
        {
            Assert.True(permutation[id] >= 0 && permutation[id] < vocabSize);
        }
    }

    [Fact]
    public void BitLinear_WithPermutation_ForwardMatchesUnpermuted()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 4, outputDimension: 3));
        layer.QuantizeFromFullPrecision(new float[,]
        {
            { 2.0f, -2.0f, 0.05f, 2.0f },
            { -2.0f, 2.0f, 2.0f, -2.0f },
            { 2.0f, 2.0f, -2.0f, 0.05f }
        });

        var input = new float[,] { { 0.5f, -0.3f, 0.7f, -0.1f } };

        // Capture output before permutation
        var outputBefore = layer.Forward(input);

        // Apply a permutation
        layer.ApplyRowPermutation([2, 0, 1]); // row 0->2, row 1->0, row 2->1

        // Output should be semantically identical (same logits, same order)
        var outputAfter = layer.Forward(input);

        Assert.Equal(outputBefore[0, 0], outputAfter[0, 0], 5);
        Assert.Equal(outputBefore[0, 1], outputAfter[0, 1], 5);
        Assert.Equal(outputBefore[0, 2], outputAfter[0, 2], 5);
    }

    [Fact]
    public void BitLinear_PermutedRows_AreCacheContiguous()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 10, outputDimension: 5));
        var weights = new float[5, 10];
        for (var r = 0; r < 5; r++)
            for (var c = 0; c < 10; c++)
                weights[r, c] = (r + c) % 3 == 0 ? 2.0f : (r + c) % 3 == 1 ? -2.0f : 0.05f;
        layer.QuantizeFromFullPrecision(weights);

        // Permute so rows 1,3,4 are adjacent physically
        layer.ApplyRowPermutation([3, 0, 4, 1, 2]); // logical 1->0, 3->1, 4->2

        var perm = layer.ExportRowPermutation();
        Assert.NotNull(perm);

        // Rows 1, 3, 4 should map to physical 0, 1, 2
        Assert.Equal(0, perm![1]);
        Assert.Equal(1, perm[3]);
        Assert.Equal(2, perm[4]);
    }

    private static (BucketRecallHeatMap, ChainBucketTable) CreateTestHeatMapAndTable()
    {
        var heatMap = new BucketRecallHeatMap(32);
        var buckets = new[]
        {
            new ChainBucket(0, [5, 10, 15], 0.9f),
            new ChainBucket(1, [3, 7, 11], 0.8f)
        };
        var table = new ChainBucketTable(buckets);

        // Simulate chain 0 accepted, then chain 1 attempted (creates transition 0->1)
        for (var i = 0; i < 3; i++)
        {
            heatMap.RecordChainAttempt(0, [5, 10, 15], 0);
            heatMap.RecordChainAccepted(0);
            heatMap.RecordChainAttempt(1, [3, 7, 11], 0); // transition 0->1 recorded internally
            heatMap.RecordChainAccepted(1);
        }

        return (heatMap, table);
    }
}
