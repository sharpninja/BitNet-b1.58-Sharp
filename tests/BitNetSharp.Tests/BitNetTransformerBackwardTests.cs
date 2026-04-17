using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

public sealed class BitNetTransformerBackwardTests
{
    private static BitNetTransformer CreateTinyTransformer(int seed = 99)
    {
        var config = new BitNetConfig(
            vocabSize: 16,
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 2,
            headCount: 2,
            maxSequenceLength: 16);

        return new BitNetTransformer(config, seed);
    }

    [Fact]
    public void Backward_PopulatesAllBitLinearMasterGradients()
    {
        var transformer = CreateTinyTransformer();

        // Initialize master weights for every BitLinear used by the model so
        // gradient accumulation is active.
        foreach (var layer in transformer.Layers)
        {
            layer.Attention.QueryProjection.InitializeMasterWeights();
            layer.Attention.KeyProjection.InitializeMasterWeights();
            layer.Attention.ValueProjection.InitializeMasterWeights();
            layer.Attention.OutputProjection.InitializeMasterWeights();
            layer.FeedForward.GateProjection.InitializeMasterWeights();
            layer.FeedForward.UpProjection.InitializeMasterWeights();
            layer.FeedForward.DownProjection.InitializeMasterWeights();
        }
        transformer.OutputHead.InitializeMasterWeights();

        var tokens = new[] { 1, 3, 5, 7 };
        var logits = transformer.Forward(tokens);

        var rng = new Random(123);
        var gradLogits = new float[logits.GetLength(0), logits.GetLength(1)];
        for (var r = 0; r < gradLogits.GetLength(0); r++)
        {
            for (var c = 0; c < gradLogits.GetLength(1); c++)
            {
                gradLogits[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        transformer.Backward(gradLogits);

        // All BitLinear master gradients should exist and contain at least
        // one non-NaN, non-Inf, non-zero entry.
        AssertGradientsPopulated(transformer.OutputHead, "OutputHead");

        for (var layerIndex = 0; layerIndex < transformer.Layers.Length; layerIndex++)
        {
            var layer = transformer.Layers[layerIndex];
            AssertGradientsPopulated(layer.Attention.QueryProjection, $"Layer[{layerIndex}].Q");
            AssertGradientsPopulated(layer.Attention.KeyProjection, $"Layer[{layerIndex}].K");
            AssertGradientsPopulated(layer.Attention.ValueProjection, $"Layer[{layerIndex}].V");
            AssertGradientsPopulated(layer.Attention.OutputProjection, $"Layer[{layerIndex}].O");
            AssertGradientsPopulated(layer.FeedForward.GateProjection, $"Layer[{layerIndex}].Gate");
            AssertGradientsPopulated(layer.FeedForward.UpProjection, $"Layer[{layerIndex}].Up");
            AssertGradientsPopulated(layer.FeedForward.DownProjection, $"Layer[{layerIndex}].Down");
        }
    }

    [Fact]
    public void Backward_ReturnsFiniteTokenEmbeddingGradients()
    {
        var transformer = CreateTinyTransformer(seed: 5);

        foreach (var layer in transformer.Layers)
        {
            layer.Attention.QueryProjection.InitializeMasterWeights();
            layer.Attention.KeyProjection.InitializeMasterWeights();
            layer.Attention.ValueProjection.InitializeMasterWeights();
            layer.Attention.OutputProjection.InitializeMasterWeights();
            layer.FeedForward.GateProjection.InitializeMasterWeights();
            layer.FeedForward.UpProjection.InitializeMasterWeights();
            layer.FeedForward.DownProjection.InitializeMasterWeights();
        }
        transformer.OutputHead.InitializeMasterWeights();

        var tokens = new[] { 2, 4 };
        var logits = transformer.Forward(tokens);

        var gradLogits = new float[logits.GetLength(0), logits.GetLength(1)];
        for (var r = 0; r < gradLogits.GetLength(0); r++)
        {
            for (var c = 0; c < gradLogits.GetLength(1); c++)
            {
                gradLogits[r, c] = 0.1f;
            }
        }

        transformer.Backward(gradLogits);

        var embGrads = transformer.ExportTokenEmbeddingGradients();
        Assert.NotNull(embGrads);

        var anyNonZero = false;
        for (var r = 0; r < embGrads!.GetLength(0); r++)
        {
            for (var c = 0; c < embGrads.GetLength(1); c++)
            {
                var v = embGrads[r, c];
                Assert.False(float.IsNaN(v), $"NaN embedding gradient at [{r},{c}]");
                Assert.False(float.IsInfinity(v), $"Inf embedding gradient at [{r},{c}]");
                if (v != 0f)
                {
                    anyNonZero = true;
                }
            }
        }

        Assert.True(anyNonZero, "Expected token-embedding gradients to contain non-zero entries after backward pass.");

        // Gradient rows for tokens NOT in the input batch should remain zero.
        var usedTokens = new HashSet<int>(tokens);
        for (var tokenId = 0; tokenId < embGrads.GetLength(0); tokenId++)
        {
            if (usedTokens.Contains(tokenId))
            {
                continue;
            }
            for (var d = 0; d < embGrads.GetLength(1); d++)
            {
                Assert.Equal(0f, embGrads[tokenId, d]);
            }
        }
    }

    [Fact]
    public void Backward_RepeatedCalls_AccumulateGradients()
    {
        var transformer = CreateTinyTransformer(seed: 8);
        transformer.OutputHead.InitializeMasterWeights();
        foreach (var layer in transformer.Layers)
        {
            layer.Attention.QueryProjection.InitializeMasterWeights();
            layer.Attention.KeyProjection.InitializeMasterWeights();
            layer.Attention.ValueProjection.InitializeMasterWeights();
            layer.Attention.OutputProjection.InitializeMasterWeights();
            layer.FeedForward.GateProjection.InitializeMasterWeights();
            layer.FeedForward.UpProjection.InitializeMasterWeights();
            layer.FeedForward.DownProjection.InitializeMasterWeights();
        }

        var tokens = new[] { 1, 2, 3 };
        var logits = transformer.Forward(tokens);
        var gradLogits = new float[logits.GetLength(0), logits.GetLength(1)];
        for (var r = 0; r < gradLogits.GetLength(0); r++)
        {
            for (var c = 0; c < gradLogits.GetLength(1); c++)
            {
                gradLogits[r, c] = 0.05f;
            }
        }

        transformer.Backward(gradLogits);
        var firstGrads = transformer.OutputHead.ExportMasterGradients();
        Assert.NotNull(firstGrads);

        // second forward/backward without zeroing — gradients should accumulate.
        _ = transformer.Forward(tokens);
        transformer.Backward(gradLogits);
        var secondGrads = transformer.OutputHead.ExportMasterGradients();
        Assert.NotNull(secondGrads);

        // At least one entry should roughly double.
        var foundAccumulation = false;
        for (var i = 0; i < firstGrads!.Length; i++)
        {
            if (MathF.Abs(firstGrads[i]) > 1e-4f
                && MathF.Abs(secondGrads![i] - 2f * firstGrads[i]) < 1e-3f)
            {
                foundAccumulation = true;
                break;
            }
        }

        Assert.True(foundAccumulation, "Expected gradients to accumulate across successive backward calls.");
    }

    private static void AssertGradientsPopulated(BitLinear layer, string name)
    {
        var grads = layer.ExportMasterGradients();
        Assert.NotNull(grads);
        var anyNonZero = false;
        foreach (var g in grads!)
        {
            Assert.False(float.IsNaN(g), $"{name}: NaN gradient");
            Assert.False(float.IsInfinity(g), $"{name}: Inf gradient");
            if (g != 0f)
            {
                anyNonZero = true;
            }
        }
        Assert.True(anyNonZero, $"{name}: expected at least one non-zero master gradient");
    }
}
