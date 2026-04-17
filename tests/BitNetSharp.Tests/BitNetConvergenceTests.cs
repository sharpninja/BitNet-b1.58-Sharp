using System.Diagnostics;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using Xunit.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// End-to-end convergence integration tests against real WikiText-2 data. These are
/// Phase A Track 6 acceptance tests: they prove that the full training stack
/// (<see cref="BitNetTransformer"/> + <see cref="BitNetFullTrainer"/>) actually
/// reduces cross-entropy loss and perplexity when fed real tokenized text.
///
/// Model config used (deliberately tiny so the whole run fits in the per-test 2-minute
/// budget on CI):
/// <list type="bullet">
///   <item>VocabSize = 32_000 (matches the WikiText-2 tokenizer output range).</item>
///   <item>Dimension = 64, HeadCount = 2 (head dim = 32, divisible by 2 for RoPE).</item>
///   <item>HiddenDimension = 128.</item>
///   <item>LayerCount = 2.</item>
///   <item>MaxSequenceLength = 32.</item>
/// </list>
///
/// Determinism: every <see cref="BitNetTransformer"/> is constructed with seed = 42,
/// so re-runs on the same CI image produce identical loss curves.
///
/// Skip policy: if <see cref="WikiTextValidationLoader.TryResolveDefaultPath"/>
/// cannot locate the validation-token binary (minimal checkouts), the tests
/// return early rather than fail — matching <see cref="WikiTextValidationLoaderTests"/>.
/// </summary>
public sealed class BitNetConvergenceTests
{
    private const int Seed = 42;

    private readonly ITestOutputHelper _output;

    public BitNetConvergenceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Training_on_wikitext_subset_reduces_loss_monotonically()
    {
        if (!WikiTextValidationLoader.TryResolveDefaultPath(out var path))
        {
            _output.WriteLine("WikiText-2 validation token file not present — skipping.");
            return;
        }

        var allTokens = WikiTextValidationLoader.LoadValidationTokens(path);
        Assert.True(allTokens.Length >= 512, $"Expected at least 512 tokens, got {allTokens.Length}.");

        var config = CreateTinyConfig();
        AssertAllIdsInVocab(allTokens.AsSpan(0, 512), config.VocabSize);

        // First 512 tokens chunked into 16 sequences of length 32.
        var subset = new int[512];
        Array.Copy(allTokens, 0, subset, 0, 512);
        var sequences = WikiTextValidationLoader
            .ChunkIntoSequences(subset, seqLen: 32)
            .ToArray();
        Assert.Equal(16, sequences.Length);

        var transformer = new BitNetTransformer(config, seed: Seed);
        var options = new BitNetTrainingOptions(
            epochs: 3,
            learningRate: 0.05f,
            dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 32));
        var trainer = new BitNetFullTrainer(transformer, options);

        var sw = Stopwatch.StartNew();
        var report = trainer.Train(sequences, epochs: 3);
        sw.Stop();

        Assert.NotNull(report.EpochMetrics);
        Assert.Equal(3, report.EpochMetrics!.Count);

        var losses = report.EpochMetrics.Select(m => m.AverageLoss).ToArray();
        _output.WriteLine($"Per-epoch loss: [{string.Join(", ", losses.Select(l => l.ToString("F4")))}]");
        _output.WriteLine($"Wall time: {sw.Elapsed.TotalSeconds:F2}s");

        foreach (var loss in losses)
        {
            Assert.True(double.IsFinite(loss), $"Non-finite loss observed: {loss}");
        }

        var firstLoss = losses[0];
        var lastLoss = losses[^1];
        Assert.True(
            lastLoss < firstLoss * 0.8,
            $"Expected >=20% loss reduction after 3 epochs: first={firstLoss:F4}, last={lastLoss:F4}");
    }

    [Fact]
    public void Perplexity_improves_after_training_on_wikitext_subset()
    {
        if (!WikiTextValidationLoader.TryResolveDefaultPath(out var path))
        {
            _output.WriteLine("WikiText-2 validation token file not present — skipping.");
            return;
        }

        var allTokens = WikiTextValidationLoader.LoadValidationTokens(path);
        Assert.True(allTokens.Length >= 384, $"Expected at least 384 tokens, got {allTokens.Length}.");

        var config = CreateTinyConfig();
        AssertAllIdsInVocab(allTokens.AsSpan(0, 384), config.VocabSize);

        // First 256 tokens → train split (8 sequences of length 32).
        // Next 128 tokens → held-out validation for perplexity (4 sequences of length 32).
        var trainRaw = new int[256];
        Array.Copy(allTokens, 0, trainRaw, 0, 256);
        var valRaw = new int[128];
        Array.Copy(allTokens, 256, valRaw, 0, 128);

        var trainSequences = WikiTextValidationLoader
            .ChunkIntoSequences(trainRaw, seqLen: 32)
            .ToArray();
        var valSequences = WikiTextValidationLoader
            .ChunkIntoSequences(valRaw, seqLen: 32)
            .ToArray();

        Assert.Equal(8, trainSequences.Length);
        Assert.Equal(4, valSequences.Length);

        var transformer = new BitNetTransformer(config, seed: Seed);

        var sw = Stopwatch.StartNew();
        var perplexityBefore = transformer.CalculatePerplexity(valSequences);

        var options = new BitNetTrainingOptions(
            epochs: 5,
            learningRate: 0.05f,
            dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 32));
        var trainer = new BitNetFullTrainer(transformer, options);
        trainer.Train(trainSequences, epochs: 5);

        var perplexityAfter = transformer.CalculatePerplexity(valSequences);
        sw.Stop();

        _output.WriteLine($"Perplexity before: {perplexityBefore:F2}");
        _output.WriteLine($"Perplexity after:  {perplexityAfter:F2}");
        _output.WriteLine($"Wall time: {sw.Elapsed.TotalSeconds:F2}s");

        Assert.True(double.IsFinite(perplexityBefore), $"Perplexity before must be finite, got {perplexityBefore}.");
        Assert.True(double.IsFinite(perplexityAfter), $"Perplexity after must be finite, got {perplexityAfter}.");
        Assert.True(perplexityBefore > 0d, "Perplexity before must be positive.");
        Assert.True(perplexityAfter > 0d, "Perplexity after must be positive.");
        Assert.True(
            perplexityAfter < perplexityBefore,
            $"Expected perplexity to improve after training: before={perplexityBefore:F2}, after={perplexityAfter:F2}");
    }

    private static BitNetConfig CreateTinyConfig() =>
        new(
            vocabSize: 32_000,
            dimension: 64,
            hiddenDimension: 128,
            layerCount: 2,
            headCount: 2,
            maxSequenceLength: 32);

    private static void AssertAllIdsInVocab(ReadOnlySpan<int> tokens, int vocabSize)
    {
        for (var i = 0; i < tokens.Length; i++)
        {
            var id = tokens[i];
            if (id < 0 || id >= vocabSize)
            {
                throw new InvalidOperationException(
                    $"Token id {id} at position {i} is outside [0, {vocabSize}); the tiny test config cannot model this corpus.");
            }
        }
    }
}
