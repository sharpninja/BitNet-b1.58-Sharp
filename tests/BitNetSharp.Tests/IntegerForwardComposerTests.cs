using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F0 (float-deletion wiring): compose the integer primitives (I3..I9)
/// against an actual <see cref="BitNetLayer"/> instance and verify the per-
/// element output stays within the integer-precision floor (5e-2) of the
/// existing float <see cref="BitNetLayer.Forward(float[,])"/> path.
///
/// This is the bridge between the standalone V1 primitive-composition test
/// and swapping the integer path into the production hot path. If this is
/// green, BitNetTransformer can grow a <c>ForwardInteger</c> overload that
/// routes every layer through <see cref="IntegerForwardComposer"/> without
/// perturbing the float training path.
/// </summary>
public sealed class IntegerForwardComposerTests
{
    [Fact]
    public void ForwardFullSeq_MatchesBitNetLayerForward_Mha_WithinIntegerTolerance()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var rng = new Random(131);
        var layer = new BitNetLayer(config, rng);
        var input = BuildMatrix(4, config.Dimension, rng);

        float[,] floatOut = layer.Forward(input);
        float[,] intOut = IntegerForwardComposer.ForwardFullSeq(layer, input);

        Assert.Equal(floatOut.GetLength(0), intOut.GetLength(0));
        Assert.Equal(floatOut.GetLength(1), intOut.GetLength(1));
        for (var r = 0; r < floatOut.GetLength(0); r++)
        {
            for (var c = 0; c < floatOut.GetLength(1); c++)
            {
                Assert.InRange(intOut[r, c] - floatOut[r, c], -5e-2f, 5e-2f);
            }
        }
    }

    [Fact]
    public void ForwardFullSeq_MatchesBitNetLayerForward_Gqa_WithinIntegerTolerance()
    {
        // GQA variant: headCount=4, kvHeadCount=2, two query heads per KV head.
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 1,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var rng = new Random(137);
        var layer = new BitNetLayer(config, rng);
        var input = BuildMatrix(4, config.Dimension, rng);

        float[,] floatOut = layer.Forward(input);
        float[,] intOut = IntegerForwardComposer.ForwardFullSeq(layer, input);

        for (var r = 0; r < floatOut.GetLength(0); r++)
        {
            for (var c = 0; c < floatOut.GetLength(1); c++)
            {
                Assert.InRange(intOut[r, c] - floatOut[r, c], -5e-2f, 5e-2f);
            }
        }
    }

    private static float[,] BuildMatrix(int rows, int cols, Random rng)
    {
        var m = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                m[r, c] = ((float)rng.NextDouble() - 0.5f) * 2f;
            }
        }
        return m;
    }
}
