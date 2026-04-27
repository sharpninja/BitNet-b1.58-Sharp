using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F2 (float-deletion wiring): full token-ids -> logits pass through
/// the integer path. Extends F1 (which stopped at pre-head hidden states)
/// by running FinalNorm and OutputHead through the integer primitives so
/// <see cref="BitNetTransformer"/>.Forward(tokenIds) has an all-integer
/// alternative returning float[,] logits.
///
/// Contract: per-element tolerance scales linearly with layer depth; the
/// argmax on the last-row logits must match the float reference because
/// softmax is monotonic (validated in V1 IntegerArgmax_OnLogits test).
/// </summary>
public sealed class IntegerTransformerForwardTests
{
    [Fact]
    public void ForwardInteger_MatchesForward_Argmax_OnLastRow_TwoLayerMha()
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
            seed: 211);
        var tokenIds = new[] { 3, 11, 7, 19 };

        float[,] floatLogits = transformer.Forward(tokenIds);
        float[,] intLogits = transformer.ForwardInteger(tokenIds);

        Assert.Equal(floatLogits.GetLength(0), intLogits.GetLength(0));
        Assert.Equal(floatLogits.GetLength(1), intLogits.GetLength(1));

        // Argmax on the last row drives token selection. Softmax is
        // monotonic, so drift can move logits but must not cross the
        // winner.
        int lastRow = floatLogits.GetLength(0) - 1;
        int vocab = floatLogits.GetLength(1);
        int floatArgmax = 0;
        int intArgmax = 0;
        float floatBest = floatLogits[lastRow, 0];
        float intBest = intLogits[lastRow, 0];
        for (int v = 1; v < vocab; v++)
        {
            if (floatLogits[lastRow, v] > floatBest)
            {
                floatBest = floatLogits[lastRow, v];
                floatArgmax = v;
            }
            if (intLogits[lastRow, v] > intBest)
            {
                intBest = intLogits[lastRow, v];
                intArgmax = v;
            }
        }
        Assert.Equal(floatArgmax, intArgmax);
    }

    [Fact]
    public void ForwardInteger_MatchesForward_PerElement_TwoLayerMha_WithinTolerance()
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
            seed: 223);
        var tokenIds = new[] { 3, 11, 7, 19 };

        float[,] floatLogits = transformer.Forward(tokenIds);
        float[,] intLogits = transformer.ForwardInteger(tokenIds);

        for (var r = 0; r < floatLogits.GetLength(0); r++)
        {
            for (var c = 0; c < floatLogits.GetLength(1); c++)
            {
                // Two stacked layers plus FinalNorm plus OutputHead:
                // compound integer-precision floor stays inside 5e-1.
                Assert.InRange(intLogits[r, c] - floatLogits[r, c], -5e-1f, 5e-1f);
            }
        }
    }
}
