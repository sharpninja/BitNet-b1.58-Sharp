using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I5: BitLinear gains a ForwardInt32 overload that emits an
/// <see cref="Int32ActivationBlock"/> (int32 accumulators + per-row float
/// scale combining activation scale with layer Gamma). Downstream integer
/// kernels consume the int values directly; the dequantised float[,] stays
/// accessible via ToFloat() for compatibility.
/// </summary>
public sealed class BitLinearForwardInt32Tests
{
    [Fact]
    public void ForwardInt32_ToFloat_MatchesForwardQuantizedFloat()
    {
        var layer = BuildLayerWithTernary();
        var rng = new Random(5);
        var input = new float[3, 4];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 4; c++)
            {
                input[r, c] = ((float)rng.NextDouble() - 0.5f) * 2f;
            }
        }

        var quant = QuantizedActivationBlock.FromFloat(input);
        var floatOutput = layer.ForwardQuantized(quant);
        var intBlock = layer.ForwardInt32(quant);
        var intAsFloat = intBlock.ToFloat();

        Assert.Equal(floatOutput.GetLength(0), intAsFloat.GetLength(0));
        Assert.Equal(floatOutput.GetLength(1), intAsFloat.GetLength(1));
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 4; c++)
            {
                Assert.InRange(intAsFloat[r, c] - floatOutput[r, c], -1e-5f, 1e-5f);
            }
        }
    }

    [Fact]
    public void ForwardInt32_ValuesAreIntegerAccumulators()
    {
        var layer = BuildLayerWithTernary();
        var input = new float[1, 4] { { 0.5f, -0.3f, 0.2f, 0.8f } };
        var quant = QuantizedActivationBlock.FromFloat(input);

        var intBlock = layer.ForwardInt32(quant);

        Assert.Equal(1, intBlock.Rows);
        Assert.Equal(4, intBlock.Cols);
        // Each value is the int ternary-dot; magnitude at most inputDim * 127 = 508
        for (var c = 0; c < 4; c++)
        {
            Assert.InRange(intBlock.Values[0, c], -508, 508);
        }
    }

    [Fact]
    public void ForwardInt32_RowScales_CombineGammaAndActivationScale()
    {
        var layer = BuildLayerWithTernary();
        var input = new float[2, 4];
        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < 4; c++) input[r, c] = (r + 1) * 0.1f;
        }
        var quant = QuantizedActivationBlock.FromFloat(input);

        var intBlock = layer.ForwardInt32(quant);

        for (var r = 0; r < 2; r++)
        {
            var expectedScale = layer.Gamma * quant.RowScales[r];
            Assert.InRange(intBlock.RowScales[r] - expectedScale, -1e-7f, 1e-7f);
        }
    }

    [Fact]
    public void ForwardInt32_RejectsMismatchedInputDim()
    {
        var layer = BuildLayerWithTernary();
        var wrong = new QuantizedActivationBlock(new sbyte[3], new float[1], rows: 1, cols: 3);

        Assert.Throws<ArgumentException>(() => layer.ForwardInt32(wrong));
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
