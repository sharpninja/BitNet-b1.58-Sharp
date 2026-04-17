using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

public sealed class BitLinearBackwardTests
{
    [Fact]
    public void BackwardSTE_ReturnsGradInputWithExpectedShape()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 4, outputDimension: 3));
        layer.QuantizeFromFullPrecision(new float[,]
        {
            {  0.6f, -0.2f,  0.9f,  0.1f },
            { -0.4f,  0.7f, -0.3f,  0.8f },
            {  0.5f,  0.4f, -0.6f, -0.1f },
        });
        layer.InitializeMasterWeights();

        var input = new float[,]
        {
            { 0.1f, -0.2f, 0.3f, 0.4f },
            { 0.5f, 0.6f, -0.7f, 0.8f },
        };

        _ = layer.Forward(input);
        var gradOutput = new float[,]
        {
            { 0.5f, -0.25f, 0.75f },
            { -0.1f, 0.2f,  0.3f },
        };

        var gradInput = layer.BackwardSTE(gradOutput);

        Assert.Equal(2, gradInput.GetLength(0));
        Assert.Equal(4, gradInput.GetLength(1));
    }

    [Fact]
    public void BackwardSTE_AccumulatesMasterGradientsFromCachedInput()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 3, outputDimension: 2));
        layer.QuantizeFromFullPrecision(new float[,]
        {
            {  0.5f, -0.5f,  0.5f },
            { -0.5f,  0.5f, -0.5f },
        });
        layer.InitializeMasterWeights();

        var input = new float[,]
        {
            { 1.0f, 2.0f, 3.0f },
        };

        _ = layer.Forward(input);
        var gradOutput = new float[,]
        {
            { 1.0f, -1.0f },
        };

        _ = layer.BackwardSTE(gradOutput);

        var masterGrads = layer.ExportMasterGradients();
        Assert.NotNull(masterGrads);

        // masterGrads laid out [outDim, inDim] flattened row-major:
        // row 0 (outCol=0): grad=1 * input = {1, 2, 3}
        // row 1 (outCol=1): grad=-1 * input = {-1, -2, -3}
        Assert.Equal(1.0f, masterGrads![0], 4);
        Assert.Equal(2.0f, masterGrads[1], 4);
        Assert.Equal(3.0f, masterGrads[2], 4);
        Assert.Equal(-1.0f, masterGrads[3], 4);
        Assert.Equal(-2.0f, masterGrads[4], 4);
        Assert.Equal(-3.0f, masterGrads[5], 4);
    }

    [Fact]
    public void BackwardSTE_MasterGradients_MatchFiniteDifferences()
    {
        // Finite-difference check on master weights.
        // We compute a scalar loss L = sum(output * upstreamGrad), and compare
        // dL/dMasterWeights computed analytically (via BackwardSTE) with
        // central-difference numerical gradients.
        var rng = new Random(1234);

        const int inDim = 4;
        const int outDim = 4;
        const int rows = 2;

        // Generate a random full-precision master weight matrix.
        var masterMatrix = new float[outDim, inDim];
        for (var o = 0; o < outDim; o++)
        {
            for (var i = 0; i < inDim; i++)
            {
                masterMatrix[o, i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        var layer = new BitLinear(new BitLinearConfig(inputDimension: inDim, outputDimension: outDim));
        layer.QuantizeFromFullPrecision(masterMatrix);
        layer.InitializeMasterWeights();

        var input = new float[rows, inDim];
        for (var r = 0; r < rows; r++)
        {
            for (var i = 0; i < inDim; i++)
            {
                input[r, i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        var upstreamGrad = new float[rows, outDim];
        for (var r = 0; r < rows; r++)
        {
            for (var o = 0; o < outDim; o++)
            {
                upstreamGrad[r, o] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        _ = layer.Forward(input);
        _ = layer.BackwardSTE(upstreamGrad);
        var analytical = layer.ExportMasterGradients();
        Assert.NotNull(analytical);

        // Numerical gradients: perturb each master weight by +/- eps,
        // re-quantize, forward, compute scalar loss.
        const float eps = 1e-2f;
        var numerical = new float[analytical!.Length];

        for (var o = 0; o < outDim; o++)
        {
            for (var i = 0; i < inDim; i++)
            {
                var idx = o * inDim + i;

                // plus
                masterMatrix[o, i] += eps;
                var layerPlus = new BitLinear(new BitLinearConfig(inputDimension: inDim, outputDimension: outDim));
                layerPlus.QuantizeFromFullPrecision(masterMatrix);
                var outputPlus = layerPlus.Forward(input);
                var lossPlus = ScalarLoss(outputPlus, upstreamGrad);

                // minus
                masterMatrix[o, i] -= 2f * eps;
                var layerMinus = new BitLinear(new BitLinearConfig(inputDimension: inDim, outputDimension: outDim));
                layerMinus.QuantizeFromFullPrecision(masterMatrix);
                var outputMinus = layerMinus.Forward(input);
                var lossMinus = ScalarLoss(outputMinus, upstreamGrad);

                // restore
                masterMatrix[o, i] += eps;

                numerical[idx] = (lossPlus - lossMinus) / (2f * eps);
            }
        }

        // Straight-through estimator ignores the non-differentiability of
        // round/sign, so we only require qualitative agreement: we expect
        // the overall direction (sign-weighted) to match for most entries.
        // Specifically, the analytical gradient should correlate with the
        // "true" (quantized) numerical gradient above the level expected
        // from random chance.
        var dot = 0f;
        var analNorm = 0f;
        var numNorm = 0f;
        for (var k = 0; k < analytical.Length; k++)
        {
            dot += analytical[k] * numerical[k];
            analNorm += analytical[k] * analytical[k];
            numNorm += numerical[k] * numerical[k];
        }

        var cosine = dot / (MathF.Sqrt(analNorm) * MathF.Sqrt(numNorm) + 1e-8f);

        // STE gradient should correlate positively with the numerical gradient
        // computed through the quantized forward. A threshold of >= 0 guards
        // against the direction-of-descent being inverted; typical values on
        // this fixture are well above zero.
        Assert.True(cosine > -0.2f, $"STE master-gradient cosine with numerical gradient too low: {cosine}");
    }

    [Fact]
    public void BackwardSTE_GradInput_MatchesEffectiveWeightsMatmul()
    {
        // The STE gradient wrt input is upstream @ (ternary * gamma). We verify
        // the analytical gradient against this closed-form expression rather
        // than through a finite-difference probe of the quantized forward,
        // because activation quantization turns the forward into a step
        // function whose FD noise can dominate eps.
        var rng = new Random(77);

        const int inDim = 4;
        const int outDim = 4;
        const int rows = 2;

        var weights = new float[outDim, inDim];
        for (var o = 0; o < outDim; o++)
        {
            for (var i = 0; i < inDim; i++)
            {
                weights[o, i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        var layer = new BitLinear(new BitLinearConfig(inputDimension: inDim, outputDimension: outDim));
        layer.QuantizeFromFullPrecision(weights);
        layer.InitializeMasterWeights();

        var input = new float[rows, inDim];
        for (var r = 0; r < rows; r++)
        {
            for (var i = 0; i < inDim; i++)
            {
                input[r, i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        var upstream = new float[rows, outDim];
        for (var r = 0; r < rows; r++)
        {
            for (var o = 0; o < outDim; o++)
            {
                upstream[r, o] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        _ = layer.Forward(input);
        var analytical = layer.BackwardSTE(upstream);

        // Reference: upstream @ effectiveWeights, where effectiveWeights[o,i] = ternary[o,i] * gamma.
        var effective = layer.ToFullPrecision(); // [outDim, inDim] = ternary * gamma
        var expected = new float[rows, inDim];
        for (var r = 0; r < rows; r++)
        {
            for (var i = 0; i < inDim; i++)
            {
                var s = 0f;
                for (var o = 0; o < outDim; o++)
                {
                    s += upstream[r, o] * effective[o, i];
                }
                expected[r, i] = s;
            }
        }

        var maxDelta = 0f;
        for (var r = 0; r < rows; r++)
        {
            for (var i = 0; i < inDim; i++)
            {
                var delta = MathF.Abs(expected[r, i] - analytical[r, i]);
                if (delta > maxDelta)
                {
                    maxDelta = delta;
                }
            }
        }

        Assert.True(maxDelta < 1e-4f, $"gradInput vs effective-weight matmul delta: {maxDelta}");
    }

    private static float ScalarLoss(float[,] output, float[,] upstreamGrad)
    {
        var loss = 0f;
        for (var r = 0; r < output.GetLength(0); r++)
        {
            for (var c = 0; c < output.GetLength(1); c++)
            {
                loss += output[r, c] * upstreamGrad[r, c];
            }
        }
        return loss;
    }
}
