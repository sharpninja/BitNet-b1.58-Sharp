using System.Text;

namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// Renders bucket recall heat map data as Mermaid diagrams for visualization
/// in markdown, GitHub, or any Mermaid-compatible renderer.
/// </summary>
public static class BucketRecallVisualizer
{
    /// <summary>
    /// Renders a Mermaid xychart-beta bar chart of the top tokens by accept count.
    /// </summary>
    public static string RenderTokenHeatMap(
        BucketRecallHeatMap heatMap,
        Func<int, string> tokenResolver,
        int maxTokens = 15)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentNullException.ThrowIfNull(tokenResolver);

        var topTokens = heatMap.GetTopTokensByAcceptCount(maxTokens);
        var sb = new StringBuilder();
        sb.AppendLine("```mermaid");
        sb.AppendLine("xychart-beta");
        sb.AppendLine("    title \"Token Recall Heat Map (by accept count)\"");

        if (topTokens.Count == 0)
        {
            sb.AppendLine("    x-axis [\"(no data)\"]");
            sb.AppendLine("    bar [0]");
        }
        else
        {
            var labels = string.Join(", ", topTokens.Select(t => $"\"{Escape(tokenResolver(t.TokenId))}\""));
            var values = string.Join(", ", topTokens.Select(static t => t.AcceptCount));
            sb.AppendLine($"    x-axis [{labels}]");
            sb.AppendLine($"    bar [{values}]");
        }

        sb.AppendLine("```");
        return sb.ToString();
    }

    /// <summary>
    /// Renders a Mermaid xychart-beta bar chart of chain recall rates.
    /// </summary>
    public static string RenderChainRecallChart(
        BucketRecallHeatMap heatMap,
        ChainBucketTable table,
        Func<int, string> tokenResolver,
        int maxChains = 20)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(tokenResolver);

        var chains = table.Buckets
            .Where(b => heatMap.GetChainAttemptCount(b.ChainId) > 0)
            .OrderByDescending(b => heatMap.GetChainRecallRate(b.ChainId))
            .Take(maxChains)
            .ToArray();

        var sb = new StringBuilder();
        sb.AppendLine("```mermaid");
        sb.AppendLine("xychart-beta");
        sb.AppendLine("    title \"Chain Recall Rate\"");

        if (chains.Length == 0)
        {
            sb.AppendLine("    x-axis [\"(no data)\"]");
            sb.AppendLine("    bar [0]");
        }
        else
        {
            var labels = string.Join(", ", chains.Select(c =>
                $"\"{Escape(FormatChainLabel(c, tokenResolver))}\""));
            var values = string.Join(", ", chains.Select(c =>
                (heatMap.GetChainRecallRate(c.ChainId) * 100).ToString("F0")));
            sb.AppendLine($"    x-axis [{labels}]");
            sb.AppendLine("    y-axis \"Recall %\"");
            sb.AppendLine($"    bar [{values}]");
        }

        sb.AppendLine("```");
        return sb.ToString();
    }

    /// <summary>
    /// Renders a Mermaid flowchart LR showing hot-path chain sequences with transition counts on edges.
    /// </summary>
    public static string RenderHotPathDiagram(
        BucketRecallHeatMap heatMap,
        ChainBucketTable table,
        Func<int, string> tokenResolver,
        int maxDepth = 5,
        int maxPaths = 5)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(tokenResolver);

        var hotPaths = heatMap.GetHotPaths(maxDepth, maxPaths);

        var sb = new StringBuilder();
        sb.AppendLine("```mermaid");
        sb.AppendLine("flowchart LR");

        if (hotPaths.Count == 0)
        {
            sb.AppendLine("    empty[\"No hot-paths detected\"]");
        }
        else
        {
            var declaredNodes = new HashSet<byte>();
            foreach (var path in hotPaths)
            {
                foreach (var chainId in path.ChainSequence)
                {
                    if (declaredNodes.Add(chainId))
                    {
                        var bucket = table.GetById(chainId);
                        var label = bucket is not null
                            ? FormatChainLabel(bucket, tokenResolver)
                            : $"chain {chainId}";
                        var accepts = heatMap.GetChainAcceptCount(chainId);
                        sb.AppendLine($"    C{chainId}[\"{Escape(label)}<br/>accepts: {accepts}\"]");
                    }
                }

                for (var i = 0; i < path.ChainSequence.Length - 1; i++)
                {
                    var from = path.ChainSequence[i];
                    var to = path.ChainSequence[i + 1];
                    var count = heatMap.GetTransitionCount(from, to);
                    sb.AppendLine($"    C{from} -->|{count}| C{to}");
                }
            }
        }

        sb.AppendLine("```");
        return sb.ToString();
    }

    /// <summary>
    /// Renders a Mermaid flowchart showing all chains with color coding:
    /// green for hot-path chains, red for low-value compaction candidates.
    /// </summary>
    public static string RenderCompactionReport(
        BucketRecallHeatMap heatMap,
        ChainBucketTable table,
        Func<int, string> tokenResolver,
        double threshold = 0.5)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(tokenResolver);

        var rankings = heatMap.RankBucketsForCompaction(table);
        var lowValue = heatMap.IdentifyLowValueBuckets(table, threshold);

        var sb = new StringBuilder();
        sb.AppendLine("```mermaid");
        sb.AppendLine("flowchart TD");

        var greenNodes = new List<string>();
        var redNodes = new List<string>();

        foreach (var ranking in rankings)
        {
            var bucket = table.GetById(ranking.ChainId);
            var label = bucket is not null
                ? FormatChainLabel(bucket, tokenResolver)
                : $"chain {ranking.ChainId}";
            var rate = ranking.AggregateRecallRate * 100;
            var nodeId = $"C{ranking.ChainId}";

            sb.AppendLine($"    {nodeId}[\"{Escape(label)}<br/>recall: {rate:F0}% accepts: {ranking.TotalAcceptCount}\"]");

            if (ranking.OnHotPath)
            {
                greenNodes.Add(nodeId);
            }
            else if (lowValue.Contains(ranking.ChainId))
            {
                redNodes.Add(nodeId);
            }
        }

        if (greenNodes.Count > 0)
        {
            sb.AppendLine($"    style {string.Join(",", greenNodes)} fill:#4c4,stroke:#393,color:#fff");
        }

        if (redNodes.Count > 0)
        {
            sb.AppendLine($"    style {string.Join(",", redNodes)} fill:#f44,stroke:#c33,color:#fff");
        }

        sb.AppendLine("```");
        return sb.ToString();
    }

    private static string FormatChainLabel(ChainBucket bucket, Func<int, string> tokenResolver)
    {
        var tokens = bucket.TokenIds.Select(tokenResolver);
        return string.Join(" ", tokens);
    }

    private static string Escape(string text) =>
        text.Replace("\"", "#quot;", StringComparison.Ordinal);
}
