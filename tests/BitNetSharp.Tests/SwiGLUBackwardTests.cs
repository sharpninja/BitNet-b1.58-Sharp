using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

public sealed class SwiGLUBackwardTests
{
    [Fact]
    public void BackwardSTE_ProducesGradientWithInputShape()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 8,
            hiddenDimension: 16,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16);

        var ffn = new SwiGLUFeedForward(config, new Random(1));

        // Initialize master weights so master-gradient accumulation is active.
        ffn.GateProjection.InitializeMasterWeights();
        ffn.UpProjection.InitializeMasterWeights();
        ffn.DownProjection.InitializeMasterWeights();

        var rng = new Random(2);
        var input = new float[3, 8];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        _ = ffn.Forward(input);
        var gradOut = new float[3, 8];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                gradOut[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        var gradIn = ffn.BackwardSTE(gradOut);
        Assert.Equal(3, gradIn.GetLength(0));
        Assert.Equal(8, gradIn.GetLength(1));

        // Expect some master gradients to populate on at least one sub-projection.
        var gateGrads = ffn.GateProjection.ExportMasterGradients();
        Assert.NotNull(gateGrads);
        var anyNonZero = false;
        foreach (var g in gateGrads!)
        {
            if (g != 0f)
            {
                anyNonZero = true;
                break;
            }
        }
        Assert.True(anyNonZero, "Expected at least one non-zero master gradient on GateProjection.");
    }
}
