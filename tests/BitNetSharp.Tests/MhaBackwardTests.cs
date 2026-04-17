using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

public sealed class MhaBackwardTests
{
    [Fact]
    public void BackwardSTE_ReturnsGradientMatchingInputShape()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16);

        var mha = new MultiHeadAttention(config, new Random(11));

        mha.QueryProjection.InitializeMasterWeights();
        mha.KeyProjection.InitializeMasterWeights();
        mha.ValueProjection.InitializeMasterWeights();
        mha.OutputProjection.InitializeMasterWeights();

        var rng = new Random(12);
        var input = new float[4, 8];
        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        _ = mha.Forward(input);

        var gradOut = new float[4, 8];
        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                gradOut[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        var gradIn = mha.BackwardSTE(gradOut);

        Assert.Equal(4, gradIn.GetLength(0));
        Assert.Equal(8, gradIn.GetLength(1));

        // Gradient flows through all four projections; at least one master
        // gradient on Q should be non-zero.
        var qGrads = mha.QueryProjection.ExportMasterGradients();
        Assert.NotNull(qGrads);
        var anyNonZero = false;
        foreach (var g in qGrads!)
        {
            if (!float.IsNaN(g) && !float.IsInfinity(g) && g != 0f)
            {
                anyNonZero = true;
                break;
            }
        }
        Assert.True(anyNonZero, "Expected Q-projection master gradient to populate.");
    }

    [Fact]
    public void BackwardSTE_ProducesFiniteGradients()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16);

        var mha = new MultiHeadAttention(config, new Random(21));
        mha.QueryProjection.InitializeMasterWeights();
        mha.KeyProjection.InitializeMasterWeights();
        mha.ValueProjection.InitializeMasterWeights();
        mha.OutputProjection.InitializeMasterWeights();

        var rng = new Random(22);
        var input = new float[3, 8];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        _ = mha.Forward(input);

        var gradOut = new float[3, 8];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                gradOut[r, c] = 0.5f;
            }
        }

        var gradIn = mha.BackwardSTE(gradOut);

        for (var r = 0; r < gradIn.GetLength(0); r++)
        {
            for (var c = 0; c < gradIn.GetLength(1); c++)
            {
                Assert.False(float.IsNaN(gradIn[r, c]), $"NaN at [{r},{c}]");
                Assert.False(float.IsInfinity(gradIn[r, c]), $"Inf at [{r},{c}]");
            }
        }
    }
}
