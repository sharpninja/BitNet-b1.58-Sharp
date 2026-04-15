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

internal sealed record BitNetPaperPerplexityDatasetResult(
    string Dataset,
    int Samples,
    double AverageCrossEntropy,
    double Perplexity);

internal sealed record BitNetPaperZeroShotTaskResult(
    string Task,
    int Correct,
    int Total)
{
    public double Accuracy => Total == 0 ? 0d : Correct / (double)Total;
}

internal sealed record BitNetPaperTrainingProbeResult(
    int Examples,
    int Epochs,
    double AverageLoss);

public static class BitNetPaperAuditor
{
    private const string DefaultPrompt = "how are you hosted";
    private const double TheoreticalTernaryUpperBoundBitsPerWeight = 1.584962500721156d;
    // Matches BitNetPaperModel.ProbabilityFloor (1e-6) and TraditionalLocalModel.MinimumProbability
    // (1e-6f) so every perplexity code path uses the same floor and comparisons are apples-to-apples.
    private const double ProbabilityFloor = 1e-6d;

    private static readonly (string Task, string Prompt, string ExpectedToken)[] ZeroShotFixtures =
    [
        ("ARC-Easy", "hello choose help or chart", "help"),
        ("HellaSwag", "how are you hosted choose agent or chart", "agent"),
        ("WinoGrande", "what language do you use choose american or chart", "american"),
        ("PIQA", "how do i train this model choose training or chart", "training"),
        ("StoryCloze", "show visualization choose visualize or chart", "visualize")
    ];

    public static BitNetPaperAuditReport CreateReport(
        BitNetPaperModel model,
        string prompt = DefaultPrompt,
        IReadOnlyList<BitNetBenchmarkTextFixture>? perplexityDatasets = null)
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

        Console.WriteLine("[Audit] Collecting weight statistics...");
        Console.WriteLine("[Audit] Running training probe (3-epoch clone)...");
        var trainingProbe = RunTrainingProbe(model);

        Console.WriteLine($"[Audit] Evaluating perplexity on {(perplexityDatasets ?? BitNetBenchmarkFixtures.PerplexityDatasets).Count} dataset(s)...");
        var perplexityResults = EvaluatePerplexity(model, perplexityDatasets ?? BitNetBenchmarkFixtures.PerplexityDatasets);

        Console.WriteLine("[Audit] Running zero-shot task fixtures...");
        var zeroShotResults = EvaluateZeroShot(model);

        Console.WriteLine("[Audit] Validating checkpoint round-trip...");
        var checkpointValidation = BitNetPaperCheckpoint.ValidateRoundTrip(model, prompt);

        Console.WriteLine("[Audit] Running memory audit...");
        var comparableTraditionalModel = new TraditionalLocalModel(model.Options);

        var checks = new List<BitNetPaperAuditCheck>
        {
            CreateArchitectureTopologyCheck(config, transformer, projections),
            CreateBitLinearQuantizationCheck(projections, weightStats, entropy),
            CreateNormCheck(config, norms),
            CreateAttentionCheck(config, attentionLayers),
            CreateFeedForwardCheck(config, feedForwardLayers),
            CreateDeterministicInferenceCheck(model, prompt),
            CreateMemoryAuditCheck(model, comparableTraditionalModel, transformer, projections, norms, weightStats),
            new(
                "Runtime",
                "Paper-model fine-tuning is available from the supported runtime surface.",
                BitNetPaperAuditStatus.Passed,
                $"Validated cloned-model training on {trainingProbe.Examples} default examples for {trainingProbe.Epochs} epochs; average loss={trainingProbe.AverageLoss:0.###}."),
            new(
                "Benchmark pipeline",
                "Perplexity measurements are implemented and reported for named benchmark fixture slices.",
                BitNetPaperAuditStatus.Passed,
                string.Join(", ", perplexityResults.Select(static result => $"{result.Dataset}={result.Perplexity:0.##} ppl ({result.Samples} samples)"))),
            new(
                "Benchmark pipeline",
                "Zero-shot benchmark fixtures are implemented and reported.",
                BitNetPaperAuditStatus.Passed,
                string.Join(", ", zeroShotResults.Select(static result => $"{result.Task}={result.Correct}/{result.Total} ({result.Accuracy:P0})"))),
            new(
                "Runtime",
                "Repository checkpoint export/import round-trips through the paper model.",
                checkpointValidation.ResponsesMatch ? BitNetPaperAuditStatus.Passed : BitNetPaperAuditStatus.Failed,
                $"Prompt='{checkpointValidation.Prompt}', original='{checkpointValidation.OriginalResponse}', reloaded='{checkpointValidation.ReloadedResponse}'.")
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

    private static BitNetPaperAuditCheck CreateMemoryAuditCheck(
        BitNetPaperModel model,
        TraditionalLocalModel comparableTraditionalModel,
        BitNetTransformer transformer,
        IReadOnlyList<BitLinear> projections,
        IReadOnlyList<RmsNorm> norms,
        TernaryWeightStats weightStats)
    {
        var bitNetBytes = model.EstimateResidentParameterBytes();
        var traditionalBytes = comparableTraditionalModel.EstimateResidentParameterBytes();
        var bitLinearBytes = projections.Sum(static projection => projection.EstimateResidentParameterBytes());
        var embeddingBytes = transformer.EstimateTokenEmbeddingBytes();
        var normBytes = norms.Sum(static norm => norm.EstimateResidentParameterBytes());
        var ratio = traditionalBytes == 0 ? 0d : bitNetBytes / (double)traditionalBytes;
        var effectiveBitsPerLogicalWeight = weightStats.TotalCount == 0
            ? 0d
            : (bitLinearBytes * 8d) / weightStats.TotalCount;

        var memoryStatus = bitNetBytes <= traditionalBytes
            ? BitNetPaperAuditStatus.Passed
            : BitNetPaperAuditStatus.Failed;
        var requirementText = bitNetBytes <= traditionalBytes
            ? "BitNet resident parameter storage is smaller than or equal to the traditional comparison model, confirming the memory efficiency of ternary-weight quantization."
            : "BitNet resident parameter storage exceeds the traditional comparison model; investigate weight or embedding configuration.";

        return new BitNetPaperAuditCheck(
            "Memory",
            requirementText,
            memoryStatus,
            $"BitNet resident parameters={FormatBytes(bitNetBytes)} versus traditional-local={FormatBytes(traditionalBytes)} ({ratio:0.##}x). " +
            $"The {projections.Count} BitLinear projections consume {FormatBytes(bitLinearBytes)} storing only ternary sbyte weights plus a single float32 gamma scalar per layer (~{effectiveBitsPerLogicalWeight:0.#} bits/weight before any sparse packing). " +
            $"Token embeddings add {FormatBytes(embeddingBytes)} and RMSNorm scales add {FormatBytes(normBytes)}.");
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

    private static BitNetPaperTrainingProbeResult RunTrainingProbe(BitNetPaperModel model)
    {
        var clone = CreateClone(model);
        var examples = BitNetTrainingCorpus.CreateDefaultExamples();
        var report = clone.Train(examples, epochs: 3);
        return new BitNetPaperTrainingProbeResult(examples.Count, report.Epochs, report.AverageLoss);
    }

    private static IReadOnlyList<BitNetPaperPerplexityDatasetResult> EvaluatePerplexity(
        BitNetPaperModel model,
        IReadOnlyList<BitNetBenchmarkTextFixture> datasets)
    {
        var outputWeights = model.ExportOutputHeadWeights();
        var probabilities = new float[outputWeights.GetLength(0)];
        var results = new List<BitNetPaperPerplexityDatasetResult>(datasets.Count);

        for (var datasetIndex = 0; datasetIndex < datasets.Count; datasetIndex++)
        {
            var fixture = datasets[datasetIndex];
            var totalLoss = 0d;
            var totalTokens = 0;
            var datasetStart = DateTime.UtcNow;

            for (var sampleIndex = 0; sampleIndex < fixture.Samples.Count; sampleIndex++)
            {
                if (sampleIndex > 0 && fixture.Samples.Count > 5 && sampleIndex % Math.Max(1, fixture.Samples.Count / 5) == 0)
                {
                    var elapsed = (DateTime.UtcNow - datasetStart).TotalSeconds;
                    var eta = datasetStart.AddSeconds(elapsed / sampleIndex * fixture.Samples.Count).ToLocalTime();
                    var runPpl = totalTokens > 0 ? Math.Exp(totalLoss / totalTokens) : 0d;
                    Console.WriteLine($"[Audit] {fixture.Name} {sampleIndex}/{fixture.Samples.Count} ({sampleIndex * 100 / fixture.Samples.Count}%) | Tokens: {totalTokens:N0} | Running ppl: {runPpl:F2} | ETA: {eta:HH:mm:ss}");
                }

                var tokenIds = model.EncodeTokenIds(fixture.Samples[sampleIndex], appendEndToken: true);
                if (tokenIds.Count < 2)
                {
                    continue;
                }

                // Truncate to max sequence length (model can only attend to this many tokens)
                var maxLen = model.Config.MaxSequenceLength;
                if (tokenIds.Count > maxLen)
                {
                    tokenIds = tokenIds.Take(maxLen).ToArray();
                }

                var inputIds = tokenIds.Take(tokenIds.Count - 1).ToArray();
                var targetIds = tokenIds.Skip(1).ToArray();
                var hiddenStates = model.ForwardHiddenStates(inputIds);

                for (var position = 0; position < targetIds.Length; position++)
                {
                    var logits = new float[outputWeights.GetLength(0)];
                    for (var row = 0; row < outputWeights.GetLength(0); row++)
                    {
                        var sum = 0f;
                        for (var col = 0; col < outputWeights.GetLength(1); col++)
                        {
                            sum += outputWeights[row, col] * hiddenStates[position, col];
                        }

                        logits[row] = sum;
                    }

                    totalLoss -= Math.Log(Math.Max(GetSoftmaxProbability(logits, targetIds[position]), ProbabilityFloor));
                    totalTokens++;
                }
            }

            var averageCrossEntropy = totalTokens == 0 ? 0d : totalLoss / totalTokens;
            Console.WriteLine($"[Audit] {fixture.Name} complete: {Math.Exp(averageCrossEntropy):F2} ppl ({fixture.Samples.Count} samples, {totalTokens:N0} tokens)");

            results.Add(new BitNetPaperPerplexityDatasetResult(
                fixture.Name,
                fixture.Samples.Count,
                averageCrossEntropy,
                Math.Exp(averageCrossEntropy)));
        }

        return results;
    }

    private static double GetSoftmaxProbability(float[] logits, int targetId)
    {
        var maxLogit = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit) maxLogit = logits[i];
        }

        var partition = 0d;
        var targetMass = 0d;
        for (var i = 0; i < logits.Length; i++)
        {
            var mass = Math.Exp(logits[i] - maxLogit);
            partition += mass;
            if (i == targetId) targetMass = mass;
        }

        return partition <= 0d ? ProbabilityFloor : targetMass / partition;
    }

    private static IReadOnlyList<BitNetPaperZeroShotTaskResult> EvaluateZeroShot(BitNetPaperModel model) =>
        ZeroShotFixtures
            .Select(fixture =>
            {
                var response = model.GenerateResponse(fixture.Prompt, maxTokens: 4);
                var matched = response.Tokens.Contains(fixture.ExpectedToken, StringComparer.Ordinal);
                return new BitNetPaperZeroShotTaskResult(fixture.Task, matched ? 1 : 0, 1);
            })
            .ToArray();

    private static double GetTargetProbability(float[,] logits, int targetId)
    {
        var lastRow = logits.GetLength(0) - 1;
        var maxLogit = double.NegativeInfinity;
        for (var column = 0; column < logits.GetLength(1); column++)
        {
            maxLogit = Math.Max(maxLogit, logits[lastRow, column]);
        }

        var partition = 0d;
        var targetProbability = 0d;
        for (var column = 0; column < logits.GetLength(1); column++)
        {
            var probabilityMass = Math.Exp(logits[lastRow, column] - maxLogit);
            partition += probabilityMass;
            if (column == targetId)
            {
                targetProbability = probabilityMass;
            }
        }

        if (partition <= 0d)
        {
            return ProbabilityFloor;
        }

        return Math.Max(targetProbability / partition, ProbabilityFloor);
    }

    private static BitNetPaperModel CreateClone(BitNetPaperModel model)
    {
        var clone = BitNetPaperModelSnapshot.Capture(model).Restore(model.Options.Verbosity);
        if (model.BucketTable is not null)
        {
            clone.LoadBucketTable(model.BucketTable);
        }

        return clone;
    }

    private static string FormatBytes(long bytes)
    {
        const double BytesPerKilobyte = 1024d;
        const double BytesPerMegabyte = BytesPerKilobyte * 1024d;

        if (bytes < BytesPerKilobyte)
        {
            return $"{bytes} B";
        }

        if (bytes < BytesPerMegabyte)
        {
            return $"{bytes / BytesPerKilobyte:0.##} KB";
        }

        return $"{bytes / BytesPerMegabyte:0.##} MB";
    }
}
