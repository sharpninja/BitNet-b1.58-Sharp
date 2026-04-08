using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Tests;

public sealed class BucketRecallVisualizerTests
{
    private const int TestVocabSize = 32;

    private static string ResolveToken(int tokenId) => $"tok_{tokenId}";

    [Fact]
    public void RenderTokenHeatMap_StartsWithMermaidXyChartBlock()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        heatMap.RecordTokenAccepted(0, 5);

        var result = BucketRecallVisualizer.RenderTokenHeatMap(heatMap, ResolveToken);

        Assert.StartsWith("```mermaid", result, StringComparison.Ordinal);
        Assert.Contains("xychart-beta", result, StringComparison.Ordinal);
        Assert.EndsWith("```", result.TrimEnd(), StringComparison.Ordinal);
    }

    [Fact]
    public void RenderTokenHeatMap_ContainsExpectedTokenLabels()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        heatMap.RecordTokenAccepted(0, 5);
        heatMap.RecordTokenAccepted(0, 5);
        heatMap.RecordTokenAccepted(0, 10);

        var result = BucketRecallVisualizer.RenderTokenHeatMap(heatMap, ResolveToken);

        Assert.Contains("tok_5", result, StringComparison.Ordinal);
        Assert.Contains("tok_10", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderTokenHeatMap_ReturnsEmptyDiagramWhenNoData()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);

        var result = BucketRecallVisualizer.RenderTokenHeatMap(heatMap, ResolveToken);

        Assert.StartsWith("```mermaid", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderChainRecallChart_StartsWithMermaidXyChartBlock()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([new ChainBucket(0, [1, 2], 1.0f)]);
        heatMap.RecordChainAttempt(0, [1, 2], 1);
        heatMap.RecordChainAccepted(0);

        var result = BucketRecallVisualizer.RenderChainRecallChart(heatMap, table, ResolveToken);

        Assert.StartsWith("```mermaid", result, StringComparison.Ordinal);
        Assert.Contains("xychart-beta", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderChainRecallChart_ContainsChainTokenLabels()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([new ChainBucket(0, [1, 2], 1.0f)]);
        heatMap.RecordChainAttempt(0, [1, 2], 1);
        heatMap.RecordChainAccepted(0);

        var result = BucketRecallVisualizer.RenderChainRecallChart(heatMap, table, ResolveToken);

        Assert.Contains("tok_1", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderHotPathDiagram_StartsWithMermaidFlowchart()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f)
        ]);

        for (var i = 0; i < 5; i++)
        {
            heatMap.RecordChainAccepted(0);
            heatMap.RecordChainAccepted(1);
            heatMap.ResetGenerationState();
        }

        var result = BucketRecallVisualizer.RenderHotPathDiagram(heatMap, table, ResolveToken);

        Assert.StartsWith("```mermaid", result, StringComparison.Ordinal);
        Assert.Contains("flowchart LR", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderHotPathDiagram_ContainsEdgesWithTransitionCounts()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f)
        ]);

        for (var i = 0; i < 3; i++)
        {
            heatMap.RecordChainAccepted(0);
            heatMap.RecordChainAccepted(1);
            heatMap.ResetGenerationState();
        }

        var result = BucketRecallVisualizer.RenderHotPathDiagram(heatMap, table, ResolveToken);

        Assert.Contains("-->|3|", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderCompactionReport_StartsWithMermaidFlowchart()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([
            new ChainBucket(0, [1, 2], 1.0f),
            new ChainBucket(1, [3, 4], 1.0f)
        ]);

        for (var i = 0; i < 5; i++)
        {
            heatMap.RecordChainAttempt(0, [1, 2], 1);
        }

        heatMap.RecordChainAccepted(0);

        var result = BucketRecallVisualizer.RenderCompactionReport(heatMap, table, ResolveToken);

        Assert.StartsWith("```mermaid", result, StringComparison.Ordinal);
        Assert.Contains("flowchart", result, StringComparison.Ordinal);
    }

    [Fact]
    public void RenderCompactionReport_ColorsLowValueNodesRed()
    {
        var heatMap = new BucketRecallHeatMap(TestVocabSize);
        var table = new ChainBucketTable([new ChainBucket(0, [1, 2], 1.0f)]);

        for (var i = 0; i < 5; i++)
        {
            heatMap.RecordChainAttempt(0, [1, 2], 1);
        }

        heatMap.RecordChainAccepted(0);

        var result = BucketRecallVisualizer.RenderCompactionReport(heatMap, table, ResolveToken, threshold: 0.5);

        Assert.Contains("fill:#f44", result, StringComparison.Ordinal);
    }
}
