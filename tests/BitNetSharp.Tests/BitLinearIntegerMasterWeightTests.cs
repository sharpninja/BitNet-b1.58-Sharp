using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase T1: verifies BitLinear's master-weight training state is backed by
/// <see cref="IntegerMasterWeightLayer"/> (bucket + delta fixed-point) rather
/// than a float[] accumulator. Float-shaped Export/Import remain as compat
/// boundaries for the existing trainer (T2 retires those).
/// </summary>
public sealed class BitLinearIntegerMasterWeightTests
{
    [Fact]
    public void InitializeMasterWeights_ExposesIntegerScaleProfile()
    {
        var layer = BuildLayerWithTernary();

        layer.InitializeMasterWeights();

        Assert.True(layer.IsTraining);
        var profile = layer.GetMasterWeightScaleProfile();
        Assert.NotNull(profile);
        Assert.True(profile!.Epsilon > 0f);
        Assert.True(float.IsFinite(profile.Epsilon));
        Assert.Equal(layer.Config.OutputDimension, profile.OutputDimension);
        Assert.Equal(layer.Config.InputDimension, profile.InputDimension);
    }

    [Fact]
    public void ExportMasterWeights_ReturnsFloatProjection_NearTernaryTimesGamma()
    {
        var layer = BuildLayerWithTernary();
        var gamma = layer.Gamma;
        layer.InitializeMasterWeights();

        var exported = layer.ExportMasterWeights();

        Assert.NotNull(exported);
        Assert.Equal(layer.Config.OutputDimension * layer.Config.InputDimension, exported!.Length);

        // Each element should round-trip within the quantisation step (Epsilon).
        var profile = layer.GetMasterWeightScaleProfile();
        var tolerance = profile!.Epsilon * 2f;

        // Reconstruct expected values by unpacking the ternary weights * Gamma.
        var fullPrecision = layer.ToFullPrecision();
        for (var row = 0; row < layer.Config.OutputDimension; row++)
        {
            for (var col = 0; col < layer.Config.InputDimension; col++)
            {
                var expected = fullPrecision[row, col]; // ternary * gamma
                var actual = exported[row * layer.Config.InputDimension + col];
                Assert.InRange(actual - expected, -tolerance, tolerance);
            }
        }
    }

    [Fact]
    public void ImportMasterWeights_RebuildsIntegerState_ExportRoundTripsWithinEpsilon()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();
        var profile = layer.GetMasterWeightScaleProfile()!;
        var tolerance = profile.Epsilon * 2f;

        var imported = new float[]
        {
             0.04f, -0.03f,  0.0f,  0.05f,
            -0.05f,  0.04f,  0.0f,  0.0f,
             0.0f,   0.0f,   0.04f, -0.04f,
             0.04f,  0.03f, -0.05f,  0.0f,
        };

        layer.ImportMasterWeights(imported);

        var exported = layer.ExportMasterWeights();
        Assert.NotNull(exported);
        for (var i = 0; i < imported.Length; i++)
        {
            Assert.InRange(exported![i] - imported[i], -tolerance, tolerance);
        }
    }

    [Fact]
    public void SyncTernaryFromMaster_UsesIntegerProjection_ProducesValidTernary()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();

        layer.SyncTernaryFromMaster();

        var stats = layer.GetTernaryStats();
        var total = stats.NegativeCount + stats.ZeroCount + stats.PositiveCount;
        Assert.Equal(layer.Config.OutputDimension * layer.Config.InputDimension, total);
    }

    [Fact]
    public void ApplyingPositiveGradientsRepeatedly_EventuallyProjectsToPositiveTernary()
    {
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();

        // Push weight[0] strongly positive via repeated integer deltas (through the
        // float compat API for T1; T2 replaces with direct int delta apply).
        var weights = layer.ExportMasterWeights()!;
        for (var step = 0; step < 500; step++)
        {
            weights[0] += 0.005f;
        }

        layer.ImportMasterWeights(weights);
        layer.SyncTernaryFromMaster();

        // Decode row 0 element 0 from packed ternary and confirm it went positive.
        var fullPrecision = layer.ToFullPrecision();
        Assert.True(fullPrecision[0, 0] > 0f,
            $"Expected weight[0,0] to project to positive ternary after sustained +gradient; got {fullPrecision[0, 0]}");
    }

    [Fact]
    public void BackwardSTE_StillAccumulatesFloatGradients()
    {
        // T1 leaves BackwardSTE's gradient accumulator as float[] (T3 scope).
        // Guards that the integer master-weight wiring did not accidentally
        // break the float gradient accumulation path.
        var layer = BuildLayerWithTernary();
        layer.InitializeMasterWeights();

        var input = new float[,]
        {
            { 0.1f, -0.2f, 0.3f, 0.4f },
            { 0.5f, 0.6f, -0.7f, 0.8f },
        };
        _ = layer.Forward(input);

        var gradOutput = new float[,]
        {
            { 0.5f, -0.25f, 0.75f, 0.1f },
            { -0.1f, 0.2f, 0.3f, -0.4f },
        };
        _ = layer.BackwardSTE(gradOutput);

        var grads = layer.ExportMasterGradients();
        Assert.NotNull(grads);
        Assert.Equal(layer.Config.OutputDimension * layer.Config.InputDimension, grads!.Length);
        // At least one gradient element should be non-zero after a non-trivial backward.
        Assert.Contains(grads, g => g != 0f);
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
