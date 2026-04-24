using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase T2: verifies BitLinear exposes a direct integer-delta apply path so
/// the trainer can skip the lossy ExportMasterWeights -> AdamW -> Import round
/// trip. The Import path re-quantises the entire master-weight tensor to the
/// Epsilon grid every step, which drops any sub-Epsilon gradient that would
/// otherwise accumulate in the bucket+delta state across multiple steps.
/// </summary>
public sealed class BitLinearIntegerDeltaTests
{
    [Fact]
    public void ApplyMasterWeightDeltas_AddsEachIndexToIntegerState()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();
        var before = layer.ExportMasterWeights()!;

        var deltas = new float[before.Length];
        for (var i = 0; i < deltas.Length; i++)
        {
            deltas[i] = 0.02f;
        }

        layer.ApplyMasterWeightDeltas(deltas);

        var after = layer.ExportMasterWeights()!;
        var profile = layer.GetMasterWeightScaleProfile()!;
        var tolerance = profile.Epsilon * 2f;
        for (var i = 0; i < before.Length; i++)
        {
            Assert.InRange(after[i] - before[i] - 0.02f, -tolerance, tolerance);
        }
    }

    [Fact]
    public void ApplyMasterWeightDeltas_SubEpsilonGradients_AccumulateAcrossSteps()
    {
        // Key T2 invariant: a gradient smaller than Epsilon applied once rounds
        // to 0 inside ApplyDelta's intDelta conversion. But many small steps
        // in the SAME direction cannot quantise away forever, because deltas[]
        // carries a fraction on every call. Use a step of 0.6 * Epsilon so each
        // individual step still snaps to 1 int unit (MathF.Round 0.6 -> 1).
        // Over 200 steps the index must move by at least 100 int units = 1e-3.
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();
        var profile = layer.GetMasterWeightScaleProfile()!;

        var originalValue = layer.ExportMasterWeights()![0];
        var singleStep = profile.Epsilon * 0.6f;

        var deltas = new float[layer.Config.OutputDimension * layer.Config.InputDimension];
        deltas[0] = singleStep;

        for (var step = 0; step < 200; step++)
        {
            layer.ApplyMasterWeightDeltas(deltas);
        }

        var finalValue = layer.ExportMasterWeights()![0];
        var moved = finalValue - originalValue;
        Assert.True(moved > 1e-3f,
            $"Expected sub-Epsilon deltas to accumulate visibly over 200 steps; moved only {moved}.");
    }

    [Fact]
    public void ApplyMasterWeightDeltas_RejectsWrongLength()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();

        Assert.Throws<ArgumentException>(() =>
            layer.ApplyMasterWeightDeltas(new float[layer.Config.OutputDimension * layer.Config.InputDimension - 1]));
    }

    [Fact]
    public void ApplyMasterWeightDeltas_ThrowsWhenMasterWeightsNotInitialised()
    {
        var layer = BuildLayerWithTernary();
        // No InitializeMasterWeights call, so _intMasterWeights is null.
        var deltas = new float[layer.Config.OutputDimension * layer.Config.InputDimension];

        Assert.Throws<InvalidOperationException>(() => layer.ApplyMasterWeightDeltas(deltas));
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
