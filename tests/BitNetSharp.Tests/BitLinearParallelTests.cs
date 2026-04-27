using System.Reflection;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

/// <summary>
/// H3 parallel column-stripe dispatch: BitLinear.ForwardQuantized +
/// BitLinear.ForwardInt32 wrap their outer-column loop in Parallel.For when
/// outDim crosses MinParallelOutDim. Test override
/// <c>TritDotDispatch.ForceSerial</c> pins both paths produce identical output.
/// </summary>
public sealed class BitLinearParallelTests
{
    private static FieldInfo ForceSerialField =>
        typeof(TritPacking).Assembly.GetType("BitNetSharp.Core.Quantization.TritDotDispatch")!
            .GetField("ForceSerial", BindingFlags.NonPublic | BindingFlags.Static)!;

    private static QuantizedActivationBlock BuildInput(int rows, int inDim, int seed)
    {
        var rng = new Random(seed);
        var input = new float[rows, inDim];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < inDim; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
        return QuantizedActivationBlock.FromFloat(input);
    }

    [Fact]
    public void Dispatcher_ForceSerial_ExistsAndDefaultsToFalse()
    {
        var field = ForceSerialField;
        Assert.NotNull(field);
        var orig = (bool)field.GetValue(null)!;
        Assert.False(orig, "ForceSerial defaults to false so production runs the parallel path.");
    }

    [Theory]
    [InlineData(1, 512, 512)]
    [InlineData(1, 4096, 4096)]
    [InlineData(1, 4096, 14336)]
    [InlineData(8, 4096, 4096)]
    [InlineData(8, 512, 512)]
    public void ForwardQuantized_ParallelMatchesSerial_AcrossShapes(int rows, int inDim, int outDim)
    {
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), new Random(411));
        var quant = BuildInput(rows, inDim, seed: 23 + rows + inDim);

        var force = ForceSerialField;
        var orig = (bool)force.GetValue(null)!;
        float[,] parallel, serial;
        try
        {
            force.SetValue(null, false);
            parallel = layer.ForwardQuantized(quant);
            force.SetValue(null, true);
            serial = layer.ForwardQuantized(quant);
        }
        finally
        {
            force.SetValue(null, orig);
        }

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < outDim; c++)
            {
                Assert.Equal(serial[r, c], parallel[r, c]);
            }
        }
    }

    [Theory]
    [InlineData(1, 512, 512)]
    [InlineData(1, 4096, 4096)]
    [InlineData(1, 4096, 14336)]
    [InlineData(8, 4096, 4096)]
    public void ForwardInt32_ParallelMatchesSerial_AcrossShapes(int rows, int inDim, int outDim)
    {
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), new Random(719));
        var quant = BuildInput(rows, inDim, seed: 91 + rows + inDim);

        var force = ForceSerialField;
        var orig = (bool)force.GetValue(null)!;
        Int32ActivationBlock parallel, serial;
        try
        {
            force.SetValue(null, false);
            parallel = layer.ForwardInt32(quant);
            force.SetValue(null, true);
            serial = layer.ForwardInt32(quant);
        }
        finally
        {
            force.SetValue(null, orig);
        }

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < outDim; c++)
            {
                Assert.Equal(serial.Values[r, c], parallel.Values[r, c]);
            }
        }
    }

    [Fact]
    public void ForwardQuantized_SmallOutDim_StillRunsSerial_AndProducesIdenticalOutput()
    {
        // outDim < MinParallelOutDim short-circuits to serial regardless of
        // dispatch flag. Parallel and serial paths must still match because
        // they share the same kernel.
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(256, 64), new Random(199));
        var quant = BuildInput(2, 256, seed: 41);

        var force = ForceSerialField;
        var orig = (bool)force.GetValue(null)!;
        float[,] parallel, serial;
        try
        {
            force.SetValue(null, false);
            parallel = layer.ForwardQuantized(quant);
            force.SetValue(null, true);
            serial = layer.ForwardQuantized(quant);
        }
        finally
        {
            force.SetValue(null, orig);
        }

        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < 64; c++)
            {
                Assert.Equal(serial[r, c], parallel[r, c]);
            }
        }
    }
}
