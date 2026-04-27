using System.Reflection;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

/// <summary>
/// G2 wire-up validation: confirm that the dispatcher inside
/// <see cref="TritPacking.TernaryDotSimdUnpacked"/> reaches every BitLinear
/// hot path (Forward + ForwardInt32) and that toggling
/// <c>TritDotDispatch.ForceGeneric</c> leaves <see cref="BitLinear"/> output
/// unchanged. If the wire-up regresses (e.g. someone forks a separate
/// kernel call site), one of these tests catches it.
/// </summary>
public sealed class BitLinearAvxWireUpTests
{
    private static FieldInfo ForceGenericField =>
        typeof(TritPacking).Assembly.GetType("BitNetSharp.Core.Quantization.TritDotDispatch")!
            .GetField("ForceGeneric", BindingFlags.NonPublic | BindingFlags.Static)!;

    private static (float[,] active, float[,] forcedGeneric) RunBoth(Func<float[,]> action)
    {
        var force = ForceGenericField;
        var orig = (bool)force.GetValue(null)!;
        try
        {
            force.SetValue(null, false);
            var active = action();
            force.SetValue(null, true);
            var generic = action();
            return (active, generic);
        }
        finally
        {
            force.SetValue(null, orig);
        }
    }

    private static float[,] RandomActivations(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var buf = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                buf[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
        return buf;
    }

    [Theory]
    [InlineData(1, 512, 512)]
    [InlineData(1, 4096, 4096)]
    [InlineData(1, 4096, 14336)]
    [InlineData(8, 512, 512)]
    [InlineData(8, 4096, 4096)]
    public void Forward_ProducesIdenticalOutput_OnAndOffAcceleratedDispatch(int rows, int inDim, int outDim)
    {
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), new Random(99));
        var input = RandomActivations(rows, inDim, seed: 17);

        var (active, generic) = RunBoth(() => layer.Forward((float[,])input.Clone()));

        Assert.Equal(generic.GetLength(0), active.GetLength(0));
        Assert.Equal(generic.GetLength(1), active.GetLength(1));
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < outDim; c++)
            {
                // The int sum is bit-exact across kernels, but the final float
                // surface picks up a single Gamma multiply per cell. Tiny
                // rounding can differ if instruction ordering differs; allow
                // 0.0 tolerance because Gamma multiply is the same float op
                // regardless of which kernel produced the int sum.
                Assert.Equal(generic[r, c], active[r, c]);
            }
        }
    }

    [Theory]
    [InlineData(1, 512, 512)]
    [InlineData(1, 4096, 4096)]
    [InlineData(1, 4096, 14336)]
    [InlineData(8, 4096, 4096)]
    public void ForwardInt32_ProducesIdenticalIntegerSum_OnAndOffAcceleratedDispatch(int rows, int inDim, int outDim)
    {
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), new Random(101));
        var input = RandomActivations(rows, inDim, seed: 23);
        var quant = QuantizedActivationBlock.FromFloat(input);

        var force = ForceGenericField;
        var orig = (bool)force.GetValue(null)!;
        Int32ActivationBlock active, generic;
        try
        {
            force.SetValue(null, false);
            active = layer.ForwardInt32(quant);
            force.SetValue(null, true);
            generic = layer.ForwardInt32(quant);
        }
        finally
        {
            force.SetValue(null, orig);
        }

        // Int32 surface: every cell must be bit-exact. No rounding tolerance
        // because the dot product is integer arithmetic end-to-end.
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < outDim; c++)
            {
                Assert.Equal(generic.Values[r, c], active.Values[r, c]);
            }
        }
    }

    [Fact]
    public void IntegerForwardComposer_FullLayer_BitExactAcrossDispatchToggle()
    {
        // Bonsai-shaped fixture (single layer): verifies the dispatcher
        // reaches every BitLinear inside the integer composer (8 BitLinears
        // per layer: Q/K/V/O + Gate/Up/Down) and that the final float[,]
        // output is identical regardless of which kernel was active.
        var config = new BitNetConfig(
            vocabSize: 256,
            dimension: 512,
            hiddenDimension: 1024,
            layerCount: 1,
            headCount: 8,
            maxSequenceLength: 32,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 2,
            ropeTheta: 10_000f);
        var layer = new BitNetLayer(config, new Random(311));
        var input = RandomActivations(rows: 4, cols: config.Dimension, seed: 47);

        var (active, generic) = RunBoth(() => IntegerForwardComposer.ForwardFullSeq(layer, (float[,])input.Clone()));

        Assert.Equal(generic.GetLength(0), active.GetLength(0));
        Assert.Equal(generic.GetLength(1), active.GetLength(1));
        for (var r = 0; r < generic.GetLength(0); r++)
        {
            for (var c = 0; c < generic.GetLength(1); c++)
            {
                Assert.Equal(generic[r, c], active[r, c]);
            }
        }
    }
}
