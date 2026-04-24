using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase T3: verifies BitLinear's gradient accumulator is an integer
/// bucket+delta layer rather than a float[]. Backward still computes
/// grad * cachedInput in float, but the accumulation is routed through
/// IntegerMasterWeightLayer.ApplyDelta so sub-Epsilon contributions
/// carry across accumulation steps instead of rounding to zero.
/// </summary>
public sealed class BitLinearIntegerGradientTests
{
    [Fact]
    public void ExportMasterGradients_ReturnsZerosAfterInitialize()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();

        var grads = layer.ExportMasterGradients();

        Assert.NotNull(grads);
        Assert.All(grads!, g => Assert.Equal(0f, g));
    }

    [Fact]
    public void BackwardSTE_AccumulatesGradientsIntoIntegerState_WithinEpsilon()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();
        var profile = layer.GetMasterWeightScaleProfile()!;
        var tolerance = profile.Epsilon * 4f;

        var input = new float[,]
        {
            { 0.1f, -0.2f, 0.3f, 0.4f },
            { 0.5f,  0.6f, -0.7f, 0.8f },
        };
        _ = layer.Forward(input);

        var gradOutput = new float[,]
        {
            { 0.5f, -0.25f, 0.75f, 0.1f },
            { -0.1f, 0.2f, 0.3f, -0.4f },
        };
        _ = layer.BackwardSTE(gradOutput);

        var grads = layer.ExportMasterGradients()!;

        // Reference: float oracle of grad accumulation for a single row.
        // For weight [outCol, inCol], grad_w = sum_over_row(gradOutput[row, outCol] * input[row, inCol]).
        var outDim = layer.Config.OutputDimension;
        var inDim = layer.Config.InputDimension;
        var rows = input.GetLength(0);
        for (var outCol = 0; outCol < outDim; outCol++)
        {
            for (var inCol = 0; inCol < inDim; inCol++)
            {
                var expected = 0f;
                for (var row = 0; row < rows; row++)
                {
                    expected += gradOutput[row, outCol] * input[row, inCol];
                }
                var actual = grads[outCol * inDim + inCol];
                Assert.InRange(actual - expected, -tolerance, tolerance);
            }
        }
    }

    [Fact]
    public void ZeroGradients_ClearsIntegerAccumulator()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();
        var input = new float[,]
        {
            { 0.1f, -0.2f, 0.3f, 0.4f },
        };
        _ = layer.Forward(input);
        var gradOutput = new float[,]
        {
            { 0.5f, -0.25f, 0.75f, 0.1f },
        };
        _ = layer.BackwardSTE(gradOutput);

        // Pre-zero: at least one nonzero gradient.
        var before = layer.ExportMasterGradients()!;
        Assert.Contains(before, g => g != 0f);

        layer.ZeroGradients();

        var after = layer.ExportMasterGradients()!;
        Assert.All(after, g => Assert.Equal(0f, g));
    }

    [Fact]
    public void BackwardSTE_SubEpsilonGradientContributions_AccumulateAcrossCalls()
    {
        // T3 invariant: within a single backward over multiple token rows the
        // int accumulator keeps fractional carry so tiny grad * input pieces
        // that would quantise to 0 if rounded independently can still sum.
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();
        var profile = layer.GetMasterWeightScaleProfile()!;

        // 200 rows each contribute roughly 0.6 * Epsilon to weight[0, 0].
        // grad * input = gradOutput[row, 0] * input[row, 0] ≈ 0.6 * Epsilon
        // with gradOutput=0.6*Epsilon and input=1.0.
        var rows = 200;
        var inDim = layer.Config.InputDimension;
        var outDim = layer.Config.OutputDimension;
        var input = new float[rows, inDim];
        var gradOutput = new float[rows, outDim];
        for (var r = 0; r < rows; r++)
        {
            input[r, 0] = 1.0f;
            gradOutput[r, 0] = profile.Epsilon * 0.6f;
        }
        _ = layer.Forward(input);
        _ = layer.BackwardSTE(gradOutput);

        var grads = layer.ExportMasterGradients()!;
        // Expected ~ 200 * 0.6 * Epsilon = 120 * Epsilon = 1.2e-3.
        var observed = grads[0 * inDim + 0];
        Assert.True(observed > 1e-3f,
            $"Expected sub-Epsilon grad contributions to accumulate to > 1e-3; got {observed}.");
    }

    private static BitLinear BuildLayerWithTernary()
    {
        var config = new BitLinearConfig(inputDimension: 4, outputDimension: 4);
        var layer = new BitLinear(config);
        layer.QuantizeFromFullPrecision(new float[,]
        {
            {  0.6f, -0.2f,  0.9f,  0.1f },
            { -0.4f,  0.7f, -0.3f,  0.8f },
            {  0.5f,  0.4f, -0.6f, -0.1f },
            {  0.2f, -0.5f,  0.1f,  0.4f },
        });
        return layer;
    }
}
