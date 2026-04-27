using System.Reflection;
using System.Runtime.Intrinsics.X86;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

/// <summary>
/// H2 packed-2-bit decode kernel: SSSE3 path replaces the scalar inner loop
/// in <c>TritPacking.SimdUnpackLayer</c>. Unpack today is ~5x the cost of the
/// AVX2 dot itself; an SSSE3 path that mask+shift+VPSHUFB-decodes 64 trits
/// per chunk closes that gap.
///
/// Equivalence is enforced bit-for-bit against the scalar oracle. The
/// dispatcher flag <c>TritDotDispatch.ForceScalarUnpack</c> lets tests pin
/// both code paths produce identical output.
/// </summary>
public sealed class TritDotPackedKernelTests
{
    private static readonly int[] TritLengths = new[]
    {
        1, 2, 4, 7, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 1024, 4096, 11008, 11009,
    };

    private static FieldInfo ForceScalarUnpackField =>
        typeof(TritPacking).Assembly.GetType("BitNetSharp.Core.Quantization.TritDotDispatch")!
            .GetField("ForceScalarUnpack", BindingFlags.NonPublic | BindingFlags.Static)!;

    private static (byte[] packed, sbyte[] trits) BuildFixture(int length, int seed)
    {
        var rnd = new Random(seed);
        var trits = new sbyte[length];
        for (var i = 0; i < length; i++)
        {
            trits[i] = (sbyte)(rnd.Next(3) - 1);
        }
        var packed = TritPacking.SimdPackLayer(trits);
        return (packed, trits);
    }

    private static sbyte[] DecodeViaForcedScalar(ReadOnlySpan<byte> packed, int totalTrits)
    {
        var force = ForceScalarUnpackField;
        var orig = (bool)force.GetValue(null)!;
        try
        {
            force.SetValue(null, true);
            var output = new sbyte[totalTrits];
            TritPacking.SimdUnpackLayer(packed, output, totalTrits);
            return output;
        }
        finally
        {
            force.SetValue(null, orig);
        }
    }

    private static sbyte[] DecodeViaActiveDispatch(ReadOnlySpan<byte> packed, int totalTrits)
    {
        var force = ForceScalarUnpackField;
        var orig = (bool)force.GetValue(null)!;
        try
        {
            force.SetValue(null, false);
            var output = new sbyte[totalTrits];
            TritPacking.SimdUnpackLayer(packed, output, totalTrits);
            return output;
        }
        finally
        {
            force.SetValue(null, orig);
        }
    }

    [Fact]
    public void Dispatcher_FlagExists_AndDefaultsToFalse()
    {
        // Sanity: the test suite needs the dispatch flag; if it is missing
        // the rest of these tests fail with a NullReferenceException that
        // hides the real cause.
        var field = ForceScalarUnpackField;
        Assert.NotNull(field);
        var orig = (bool)field.GetValue(null)!;
        Assert.False(orig, "ForceScalarUnpack defaults to false so production runs the SIMD path.");
    }

    [Fact]
    public void FastUnpack_MatchesScalar_AcrossLengths()
    {
        foreach (var len in TritLengths)
        {
            var (packed, _) = BuildFixture(len, seed: 100 + len);
            var scalar = DecodeViaForcedScalar(packed, len);
            var fast = DecodeViaActiveDispatch(packed, len);
            Assert.Equal(scalar.Length, fast.Length);
            for (var i = 0; i < len; i++)
            {
                Assert.Equal(scalar[i], fast[i]);
            }
        }
    }

    [Fact]
    public void FastUnpack_MatchesScalar_OnAllSinglePackedByteValues()
    {
        // Every packed byte ∈ [0, 255]: decode 4 trits via both paths and
        // confirm identical output. Includes the 0b10 code which is invalid
        // for trit encoding but the legacy scalar produces -2 for it; the
        // SIMD path must match exactly.
        for (var b = 0; b < 256; b++)
        {
            var packed = new byte[] { (byte)b };
            var scalar = DecodeViaForcedScalar(packed, 4);
            var fast = DecodeViaActiveDispatch(packed, 4);
            for (var slot = 0; slot < 4; slot++)
            {
                Assert.Equal(scalar[slot], fast[slot]);
            }
        }
    }

    [Fact]
    public void FastUnpack_MatchesScalar_OnAllPossibleQuintupleByteSeeds()
    {
        // Stress: 5 sequential packed bytes (= 20 trits, exercises the
        // chunk boundary at length 16). Sample seed-driven random patterns.
        for (var seed = 1; seed <= 20; seed++)
        {
            var (packed, _) = BuildFixture(20, seed * 13);
            var scalar = DecodeViaForcedScalar(packed, 20);
            var fast = DecodeViaActiveDispatch(packed, 20);
            for (var i = 0; i < 20; i++)
            {
                Assert.Equal(scalar[i], fast[i]);
            }
        }
    }

    [Theory]
    [InlineData(1, 512, 512)]
    [InlineData(1, 4096, 4096)]
    [InlineData(1, 4096, 14336)]
    [InlineData(8, 512, 512)]
    [InlineData(8, 4096, 4096)]
    public void BitLinear_FastUnpackPath_MatchesScalarPath(int rows, int inDim, int outDim)
    {
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), new Random(311));
        var rng = new Random(13 + rows + inDim);
        var input = new float[rows, inDim];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < inDim; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
        var quant = QuantizedActivationBlock.FromFloat(input);

        var force = ForceScalarUnpackField;
        var orig = (bool)force.GetValue(null)!;
        float[,] active, scalar;
        try
        {
            force.SetValue(null, false);
            active = layer.ForwardQuantized(quant);
            force.SetValue(null, true);
            scalar = layer.ForwardQuantized(quant);
        }
        finally
        {
            force.SetValue(null, orig);
        }

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < outDim; c++)
            {
                Assert.Equal(scalar[r, c], active[r, c]);
            }
        }
    }

    [Theory]
    [InlineData(1, 512, 512)]
    [InlineData(1, 4096, 4096)]
    [InlineData(1, 4096, 14336)]
    [InlineData(8, 4096, 4096)]
    public void BitLinear_ForwardInt32_FastUnpackPath_MatchesScalarPath(int rows, int inDim, int outDim)
    {
        var layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(inDim, outDim), new Random(641));
        var rng = new Random(73 + rows + inDim);
        var input = new float[rows, inDim];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < inDim; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }
        var quant = QuantizedActivationBlock.FromFloat(input);

        var force = ForceScalarUnpackField;
        var orig = (bool)force.GetValue(null)!;
        Int32ActivationBlock active, scalar;
        try
        {
            force.SetValue(null, false);
            active = layer.ForwardInt32(quant);
            force.SetValue(null, true);
            scalar = layer.ForwardInt32(quant);
        }
        finally
        {
            force.SetValue(null, orig);
        }

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < outDim; c++)
            {
                Assert.Equal(scalar.Values[r, c], active.Values[r, c]);
            }
        }
    }
}
