using System;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Converter.Tests;

public sealed class BitNetLayerAttentionDispatchTests
{
    [Fact]
    public void Attention_ForMhaConfig_IsMultiHeadAttention()
    {
        var config = new BitNetConfig(
            vocabSize: 1024,
            dimension: 64,
            hiddenDimension: 128,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 8,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 4,
            ropeTheta: 10_000f);

        var layer = new BitNetLayer(config, new Random(0));

        Assert.IsType<MultiHeadAttention>(layer.Attention);
    }

    [Fact]
    public void Attention_ForGqaConfig_IsGroupedQueryAttention()
    {
        var config = new BitNetConfig(
            vocabSize: 1024,
            dimension: 64,
            hiddenDimension: 128,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 8,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 2,
            ropeTheta: 1_000_000f);

        var layer = new BitNetLayer(config, new Random(0));

        Assert.IsType<GroupedQueryAttention>(layer.Attention);
    }

    [Fact]
    public void Attention_ExposesBitLinearProjectionsForBothPaths()
    {
        // Both MHA and GQA must expose Q/K/V/Output projections for the
        // trainer / audit / model parameter iteration.
        var mhaConfig = new BitNetConfig(
            vocabSize: 1024, dimension: 64, hiddenDimension: 128,
            layerCount: 2, headCount: 4, maxSequenceLength: 8,
            rmsNormEpsilon: 1e-5f, kvHeadCount: 4, ropeTheta: 10_000f);
        var gqaConfig = new BitNetConfig(
            vocabSize: 1024, dimension: 64, hiddenDimension: 128,
            layerCount: 2, headCount: 4, maxSequenceLength: 8,
            rmsNormEpsilon: 1e-5f, kvHeadCount: 2, ropeTheta: 1_000_000f);

        var mhaLayer = new BitNetLayer(mhaConfig, new Random(1));
        var gqaLayer = new BitNetLayer(gqaConfig, new Random(1));

        Assert.NotNull(mhaLayer.Attention.QueryProjection);
        Assert.NotNull(mhaLayer.Attention.KeyProjection);
        Assert.NotNull(mhaLayer.Attention.ValueProjection);
        Assert.NotNull(mhaLayer.Attention.OutputProjection);

        Assert.NotNull(gqaLayer.Attention.QueryProjection);
        Assert.NotNull(gqaLayer.Attention.KeyProjection);
        Assert.NotNull(gqaLayer.Attention.ValueProjection);
        Assert.NotNull(gqaLayer.Attention.OutputProjection);

        // GQA K/V output dim should be kvHeadCount * headDim, smaller than Q.
        Assert.Equal(mhaConfig.Dimension, mhaLayer.Attention.KeyProjection.Config.OutputDimension);
        Assert.Equal(gqaConfig.KvHeadCount * gqaConfig.HeadDimension, gqaLayer.Attention.KeyProjection.Config.OutputDimension);
    }

    [Fact]
    public void Forward_OnGqaLayer_ProducesExpectedShape()
    {
        var config = new BitNetConfig(
            vocabSize: 1024, dimension: 64, hiddenDimension: 128,
            layerCount: 2, headCount: 4, maxSequenceLength: 8,
            rmsNormEpsilon: 1e-5f, kvHeadCount: 2, ropeTheta: 1_000_000f);

        var layer = new BitNetLayer(config, new Random(2));
        var input = new float[3, 64];
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 64; c++)
            {
                input[r, c] = 0.1f * (r + 1) * MathF.Sin(c * 0.05f);
            }
        }

        var output = layer.Forward(input);
        Assert.Equal(3, output.GetLength(0));
        Assert.Equal(64, output.GetLength(1));
    }
}
