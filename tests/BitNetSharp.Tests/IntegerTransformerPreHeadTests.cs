using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F1 (float-deletion wiring): stack multiple layers through
/// <see cref="IntegerForwardComposer.ForwardFullSeq"/> inside a real
/// <see cref="BitNetTransformer"/> and verify the pre-head hidden states
/// track the float reference within a compound-drift budget.
///
/// Per-layer drift is bounded by the I3..I9 primitives at 5e-2; compounded
/// across two stacked layers the empirical worst case is still well under
/// 2.5e-1 on small shapes. If this test is green, the layer loop inside
/// <c>BitNetTransformer.Forward</c> can be swapped to the integer path
/// without touching embedding or the output head.
/// </summary>
public sealed class IntegerTransformerPreHeadTests
{
    [Fact]
    public void ForwardPreHeadIntegerStates_MatchesPreHeadStates_TwoLayerConfig()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 2,
            headCount: 2,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var transformer = new BitNetTransformer(
            config,
            NullLogger<BitNetTransformer>.Instance,
            seed: 151);
        var tokenIds = new[] { 3, 11, 7, 19 };

        float[,] floatOut = transformer.ForwardPreHeadStates(tokenIds);
        float[,] intOut = transformer.ForwardPreHeadIntegerStates(tokenIds);

        Assert.Equal(floatOut.GetLength(0), intOut.GetLength(0));
        Assert.Equal(floatOut.GetLength(1), intOut.GetLength(1));
        for (var r = 0; r < floatOut.GetLength(0); r++)
        {
            for (var c = 0; c < floatOut.GetLength(1); c++)
            {
                Assert.InRange(intOut[r, c] - floatOut[r, c], -2.5e-1f, 2.5e-1f);
            }
        }
    }

    [Fact]
    public void ForwardPreHeadIntegerStates_MatchesPreHeadStates_FourLayerGqa()
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 4,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        var transformer = new BitNetTransformer(
            config,
            NullLogger<BitNetTransformer>.Instance,
            seed: 157);
        var tokenIds = new[] { 5, 13, 17, 23 };

        float[,] floatOut = transformer.ForwardPreHeadStates(tokenIds);
        float[,] intOut = transformer.ForwardPreHeadIntegerStates(tokenIds);

        for (var r = 0; r < floatOut.GetLength(0); r++)
        {
            for (var c = 0; c < floatOut.GetLength(1); c++)
            {
                // 4 compounded layers: bump tolerance linearly in depth.
                Assert.InRange(intOut[r, c] - floatOut[r, c], -5e-1f, 5e-1f);
            }
        }
    }
}
