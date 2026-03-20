using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core;

public enum BitNetPaperAuditStatus
{
    Passed,
    Pending,
    Failed
}

public sealed record BitNetPaperAuditCheck(
    string Area,
    string Requirement,
    BitNetPaperAuditStatus Status,
    string Details);

public sealed record BitNetPaperAuditReport(
    string ModelId,
    string DisplayName,
    IReadOnlyList<BitNetPaperAuditCheck> Checks)
{
    public int PassedCount => Checks.Count(static check => check.Status == BitNetPaperAuditStatus.Passed);

    public int PendingCount => Checks.Count(static check => check.Status == BitNetPaperAuditStatus.Pending);

    public int FailedCount => Checks.Count(static check => check.Status == BitNetPaperAuditStatus.Failed);

    public bool ArchitectureChecksPassed => Checks
        .Where(static check => string.Equals(check.Area, "Architecture", StringComparison.Ordinal))
        .All(static check => check.Status == BitNetPaperAuditStatus.Passed);
}

public static class BitNetPaperAuditor
{
    private const string DefaultPrompt = "how are you hosted";
    private const double TheoreticalTernaryUpperBoundBitsPerWeight = 1.584962500721156d;

    public static BitNetPaperAuditReport CreateReport(BitNetPaperModel model, string prompt = DefaultPrompt)
    {
        ArgumentNullException.ThrowIfNull(model);

        var transformer = model.Transformer;
        var config = model.Config;
        var projections = EnumerateBitLinearLayers(transformer).ToArray();
        var norms = EnumerateNormLayers(transformer).ToArray();
        var attentionLayers = transformer.Layers.Select(static layer => layer.Attention).ToArray();
        var feedForwardLayers = transformer.Layers.Select(static layer => layer.FeedForward).ToArray();
        var weightStats = model.GetTernaryWeightStats();
        var entropy = CalculateTernaryEntropy(weightStats);

        var checks = new List<BitNetPaperAuditCheck>
        {
            CreateArchitectureTopologyCheck(config, transformer, projections),
            CreateBitLinearQuantizationCheck(projections, weightStats, entropy),
            CreateNormCheck(config, norms),
            CreateAttentionCheck(config, attentionLayers),
            CreateFeedForwardCheck(config, feedForwardLayers),
            CreateDeterministicInferenceCheck(model, prompt),
            new(
                "Roadmap",
                "Paper-aligned training loop and STE-backed optimization are available from the supported runtime surface.",
                BitNetPaperAuditStatus.Pending,
                "The active hosted BitNet runtime remains inference-only. The CLI still reports that the paper-aligned training loop is not implemented in this branch."),
            new(
                "Roadmap",
                "Perplexity parity against the paper datasets is measured in-repository.",
                BitNetPaperAuditStatus.Pending,
                "The repository does not yet run WikiText2, C4, or RedPajama perplexity evaluation in the active toolchain."),
            new(
                "Roadmap",
                "Zero-shot paper benchmark tasks are implemented and reported.",
                BitNetPaperAuditStatus.Pending,
                "ARC-Easy, HellaSwag, WinoGrande, PIQA, and StoryCloze evaluation are not yet wired into the repository runtime or reports."),
            new(
                "Roadmap",
                "Checkpoint export or interoperability with bitnet.cpp/llama.cpp is implemented.",
                BitNetPaperAuditStatus.Pending,
                "The repository does not yet export or validate a checkpoint format against an external BitNet runtime.")
        };

        return new BitNetPaperAuditReport(
            model.ModelId,
            "Paper-aligned BitNet b1.58 audit",
            checks);
    }

    private static BitNetPaperAuditCheck CreateArchitectureTopologyCheck(BitNetConfig config, BitNetTransformer transformer, IReadOnlyList<BitLinear> projections)
    {
        var expectedProjectionCount = (config.LayerCount * 7) + 1;
        var projectionCountMatches = projections.Count == expectedProjectionCount;
        var outputHeadMatches = transformer.OutputHead.Config.InputDimension == config.Dimension
            && transformer.OutputHead.Config.OutputDimension == config.VocabSize;

        var passed = projectionCountMatches
            && transformer.Layers.Length == config.LayerCount
            && outputHeadMatches;

        return new BitNetPaperAuditCheck(
            "Architecture",
            "Decoder-only transformer topology matches the paper-aligned BitNet surface.",
            passed ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
            $"Layers={transformer.Layers.Length}/{config.LayerCount}, BitLinear projections={projections.Count}/{expectedProjectionCount}, output head={transformer.OutputHead.Config.InputDimension}->{transformer.OutputHead.Config.OutputDimension}.");
    }

    private static BitNetPaperAuditCheck CreateBitLinearQuantizationCheck(
        IReadOnlyList<BitLinear> projections,
        TernaryWeightStats weightStats,
        double entropy)
    {
        var allBiasFree = projections.All(static projection => !projection.HasBias);
        var usesEightBitActivationQuantization = projections.All(static projection => projection.ActivationQuantizationBitWidth == 8 && projection.ActivationQuantizationBound == 127);
        var weightCountsMatch = weightStats.TotalCount == projections.Sum(static projection => projection.Config.InputDimension * projection.Config.OutputDimension);
        var passed = allBiasFree && usesEightBitActivationQuantization && weightCountsMatch;

        return new BitNetPaperAuditCheck(
            "Architecture",
            "BitLinear projections stay ternary, bias-free, and use signed 8-bit activation quantization.",
            passed ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
            $"Bias-free projections={allBiasFree}, activation quantization=±{projections[0].ActivationQuantizationBound} ({projections[0].ActivationQuantizationBitWidth}-bit), ternary weights={weightStats.NegativeCount}/{weightStats.ZeroCount}/{weightStats.PositiveCount}, empirical entropy={entropy:0.###} bits/weight, theoretical ternary limit={TheoreticalTernaryUpperBoundBitsPerWeight:0.###}.");
    }

    private static BitNetPaperAuditCheck CreateNormCheck(BitNetConfig config, IReadOnlyList<RmsNorm> norms)
    {
        var epsilonMatches = norms.All(norm => norm.Epsilon == config.RmsNormEpsilon && norm.Epsilon == 1e-5f);
        var biasFree = norms.All(static norm => !norm.HasBias && norm.HasLearnableScale);
        var passed = epsilonMatches && biasFree;

        return new BitNetPaperAuditCheck(
            "Architecture",
            "RMSNorm layers stay bias-free and use the paper epsilon.",
            passed ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
            $"Norm count={norms.Count}, epsilon={config.RmsNormEpsilon:0.#####}, learnable scale only={biasFree}.");
    }

    private static BitNetPaperAuditCheck CreateAttentionCheck(BitNetConfig config, IReadOnlyList<MultiHeadAttention> attentionLayers)
    {
        var dimensionsMatch = attentionLayers.All(attention =>
            attention.QueryProjection.Config.InputDimension == config.Dimension
            && attention.QueryProjection.Config.OutputDimension == config.Dimension
            && attention.KeyProjection.Config.InputDimension == config.Dimension
            && attention.KeyProjection.Config.OutputDimension == config.Dimension
            && attention.ValueProjection.Config.InputDimension == config.Dimension
            && attention.ValueProjection.Config.OutputDimension == config.Dimension
            && attention.OutputProjection.Config.InputDimension == config.Dimension
            && attention.OutputProjection.Config.OutputDimension == config.Dimension);
        var ropeAndCausalityMatch = attentionLayers.All(static attention =>
            attention.UsesRotaryPositionEmbedding
            && attention.AppliesRotaryPositionEmbeddingToQueriesAndKeysOnly
            && attention.UsesCausalAttentionMask);
        var passed = dimensionsMatch && ropeAndCausalityMatch;

        return new BitNetPaperAuditCheck(
            "Architecture",
            "Attention uses Q/K/V/O BitLinear projections, RoPE on Q/K, and causal masking.",
            passed ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
            $"Attention layers={attentionLayers.Count}, head count={config.HeadCount}, head dimension={config.HeadDimension}, scaled-dot-product factor={attentionLayers.FirstOrDefault()?.AttentionScale:0.####}.");
    }

    private static BitNetPaperAuditCheck CreateFeedForwardCheck(BitNetConfig config, IReadOnlyList<SwiGLUFeedForward> feedForwardLayers)
    {
        var dimensionsMatch = feedForwardLayers.All(feedForward =>
            feedForward.GateProjection.Config.InputDimension == config.Dimension
            && feedForward.GateProjection.Config.OutputDimension == config.HiddenDimension
            && feedForward.UpProjection.Config.InputDimension == config.Dimension
            && feedForward.UpProjection.Config.OutputDimension == config.HiddenDimension
            && feedForward.DownProjection.Config.InputDimension == config.HiddenDimension
            && feedForward.DownProjection.Config.OutputDimension == config.Dimension);
        var passed = dimensionsMatch && feedForwardLayers.All(static feedForward => feedForward.UsesSwiGLUActivation);

        return new BitNetPaperAuditCheck(
            "Architecture",
            "Feed-forward blocks use paper-style SwiGLU gate/up/down BitLinear projections.",
            passed ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
            $"Feed-forward layers={feedForwardLayers.Count}, hidden dimension={config.HiddenDimension}, SwiGLU activation={feedForwardLayers.All(static feedForward => feedForward.UsesSwiGLUActivation)}.");
    }

    private static BitNetPaperAuditCheck CreateDeterministicInferenceCheck(BitNetPaperModel model, string prompt)
    {
        var first = model.GenerateResponse(prompt, maxTokens: 4);
        var second = model.GenerateResponse(prompt, maxTokens: 4);
        var passed = string.Equals(first.ResponseText, second.ResponseText, StringComparison.Ordinal)
            && first.Tokens.SequenceEqual(second.Tokens, StringComparer.Ordinal);

        return new BitNetPaperAuditCheck(
            "Architecture",
            "Seeded inference is deterministic for repeated prompts.",
            passed ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
            $"Prompt='{prompt}', first response='{first.ResponseText}', second response='{second.ResponseText}'.");
    }

    private static IEnumerable<BitLinear> EnumerateBitLinearLayers(BitNetTransformer transformer)
    {
        foreach (var layer in transformer.Layers)
        {
            yield return layer.Attention.QueryProjection;
            yield return layer.Attention.KeyProjection;
            yield return layer.Attention.ValueProjection;
            yield return layer.Attention.OutputProjection;
            yield return layer.FeedForward.GateProjection;
            yield return layer.FeedForward.UpProjection;
            yield return layer.FeedForward.DownProjection;
        }

        yield return transformer.OutputHead;
    }

    private static IEnumerable<RmsNorm> EnumerateNormLayers(BitNetTransformer transformer)
    {
        foreach (var layer in transformer.Layers)
        {
            yield return layer.PreAttentionNorm;
            yield return layer.PreFeedForwardNorm;
        }

        yield return transformer.FinalNorm;
    }

    private static double CalculateTernaryEntropy(TernaryWeightStats stats)
    {
        if (stats.TotalCount == 0)
        {
            return 0d;
        }

        return CalculateEntropy(stats.NegativeCount, stats.TotalCount)
            + CalculateEntropy(stats.ZeroCount, stats.TotalCount)
            + CalculateEntropy(stats.PositiveCount, stats.TotalCount);
    }

    private static double CalculateEntropy(int count, int total)
    {
        if (count == 0 || total == 0)
        {
            return 0d;
        }

        var probability = count / (double)total;
        return -probability * Math.Log2(probability);
    }
}
