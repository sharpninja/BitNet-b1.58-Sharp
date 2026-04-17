using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class PerplexityTests
{
    [Fact]
    public void Perplexity_on_uniform_logits_equals_vocab_size()
    {
        // For a uniform distribution over V classes, p(target) = 1/V, NLL = ln(V),
        // and perplexity = exp(NLL) = V exactly.
        const int vocab = 32;
        const int tokenCount = 16;

        var uniformLogits = new float[tokenCount, vocab]; // all zeros -> uniform softmax

        // There are (tokenCount - 1) prediction positions per chunk.
        // Targets can be anything within [0, vocab).
        var targets = Enumerable.Range(0, tokenCount).Select(i => i % vocab).ToArray();

        var perplexity = BitNetTransformer.PerplexityFromLogits(uniformLogits, targets);

        Assert.InRange(perplexity, vocab * 0.99, vocab * 1.01);
    }

    [Fact]
    public void Perplexity_on_perfect_prediction_equals_one()
    {
        // Craft logits where the target position's logit is very high, so softmax
        // approaches a one-hot distribution and NLL -> 0, perplexity -> 1.
        const int vocab = 16;
        const int tokenCount = 8;

        var logits = new float[tokenCount, vocab];
        // Row r predicts tokens[r + 1]; set the target column far above the rest.
        var targets = new[] { 0, 1, 2, 3, 4, 5, 6, 7 };
        for (var row = 0; row < tokenCount - 1; row++)
        {
            logits[row, targets[row + 1]] = 50f;
        }

        var perplexity = BitNetTransformer.PerplexityFromLogits(logits, targets);

        Assert.InRange(perplexity, 1.0, 1.01);
    }

    [Fact]
    public void CalculatePerplexity_untrained_model_on_random_sequences_is_finite_and_reasonable()
    {
        var config = new BitNetConfig(
            vocabSize: 64,
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16);
        var model = new BitNetTransformer(config, seed: 7);

        var rng = new Random(123);
        var sequences = Enumerable.Range(0, 3)
            .Select(_ => Enumerable.Range(0, 10).Select(_ => rng.Next(0, config.VocabSize)).ToArray())
            .ToArray();

        var perplexity = model.CalculatePerplexity(sequences);

        Assert.True(double.IsFinite(perplexity));
        Assert.True(perplexity > 1.0, $"Perplexity should exceed 1 for an untrained model, got {perplexity}.");
        // An untrained model on a V=64 vocab should land near V; allow generous slack.
        Assert.True(perplexity < config.VocabSize * 4, $"Perplexity {perplexity} far exceeds reasonable bound for V={config.VocabSize}.");
    }

    [Fact]
    public void CalculatePerplexity_skips_sequences_shorter_than_two_tokens()
    {
        var config = new BitNetConfig(
            vocabSize: 16,
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 8);
        var model = new BitNetTransformer(config, seed: 1);

        // All sequences too short -> no tokens contribute -> returns 0 by convention.
        var perplexity = model.CalculatePerplexity(new[] { new[] { 1 }, Array.Empty<int>() });

        Assert.Equal(0d, perplexity);
    }

    [Fact]
    public void Perplexity_on_wikitext_validation_is_finite()
    {
        if (!WikiTextValidationLoader.TryResolveDefaultPath(out var path))
        {
            return; // data file not available in this build context
        }

        var tokens = WikiTextValidationLoader.LoadValidationTokens(path);

        // Cap vocab size so each token id fits.
        var maxId = 0;
        for (var i = 0; i < tokens.Length && i < 4096; i++)
        {
            if (tokens[i] > maxId)
            {
                maxId = tokens[i];
            }
        }

        var config = new BitNetConfig(
            vocabSize: Math.Max(maxId + 1, 2048),
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 32);
        var model = new BitNetTransformer(config, seed: 3);

        // Only take a handful of short chunks to keep the test fast.
        var sample = tokens.Take(128).Where(id => id >= 0 && id < config.VocabSize).ToArray();
        var sequences = WikiTextValidationLoader.ChunkIntoSequences(sample, seqLen: 16).Take(2).ToArray();

        if (sequences.Length == 0)
        {
            return;
        }

        var perplexity = model.CalculatePerplexity(sequences);

        Assert.True(double.IsFinite(perplexity));
        Assert.True(perplexity > 1.0, $"Expected perplexity > 1 for an untrained model, got {perplexity}.");
    }
}
