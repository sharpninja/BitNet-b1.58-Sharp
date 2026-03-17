using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

public sealed class BitLinearTests
{
    [Fact]
    public void QuantizeFromFullPrecision_ComputesAbsMeanAndTernaryStatistics()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 3, outputDimension: 2));
        var weights = new float[,]
        {
            { -2.0f, -0.4f, 0.2f },
            { 0.6f, 1.2f, -1.8f }
        };

        layer.QuantizeFromFullPrecision(weights);
        var stats = layer.GetTernaryStats();

        Assert.Equal(6.2f / 6f, layer.Gamma, 4);
        Assert.Equal(2, stats.NegativeCount);
        Assert.Equal(2, stats.ZeroCount);
        Assert.Equal(2, stats.PositiveCount);
        Assert.Equal(6, stats.TotalCount);
    }

    [Fact]
    public void ToFullPrecision_ReturnsGammaScaledTernaryWeights()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 3, outputDimension: 2));
        layer.QuantizeFromFullPrecision(new float[,]
        {
            { -2.0f, -0.4f, 0.2f },
            { 0.6f, 1.2f, -1.8f }
        });

        var restored = layer.ToFullPrecision();

        Assert.Equal(-layer.Gamma, restored[0, 0], 4);
        Assert.Equal(0f, restored[0, 1], 4);
        Assert.Equal(0f, restored[0, 2], 4);
        Assert.Equal(layer.Gamma, restored[1, 0], 4);
        Assert.Equal(layer.Gamma, restored[1, 1], 4);
        Assert.Equal(-layer.Gamma, restored[1, 2], 4);
    }

    [Fact]
    public void Forward_UsesQuantizedActivationsAndTernaryWeights()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 2, outputDimension: 1));
        layer.QuantizeFromFullPrecision(new float[,]
        {
            { 2.0f, -2.0f }
        });

        var input = new float[,]
        {
            { 0.5f, -0.25f }
        };

        var output = layer.Forward(new float[,]
        {
            { input[0, 0], input[0, 1] }
        });

        var maxAbs = MathF.Max(MathF.Abs(input[0, 0]), MathF.Abs(input[0, 1]));
        var scale = maxAbs / 127f;
        var quantizedFirst = Math.Clamp((int)MathF.Round(input[0, 0] / scale, MidpointRounding.AwayFromZero), -127, 127) * scale;
        var quantizedSecond = Math.Clamp((int)MathF.Round(input[0, 1] / scale, MidpointRounding.AwayFromZero), -127, 127) * scale;
        var expected = (quantizedFirst - quantizedSecond) * layer.Gamma;

        Assert.Equal(expected, output[0, 0], 5);
    }

    [Fact]
    public void BackwardSte_ReturnsClonedGradient()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 2, outputDimension: 1));
        var gradient = new float[,]
        {
            { 1.5f, -0.25f }
        };

        var result = layer.BackwardSTE(gradient);

        Assert.NotSame(gradient, result);
        Assert.Equal(gradient[0, 0], result[0, 0]);
        Assert.Equal(gradient[0, 1], result[0, 1]);
    }
}
