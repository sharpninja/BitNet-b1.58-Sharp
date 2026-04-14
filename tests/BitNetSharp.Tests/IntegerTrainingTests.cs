using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class IntegerTrainingTests
{
    [Fact]
    public void LayerScaleProfile_CalibratesFromGradientData()
    {
        var gradients = Enumerable.Range(0, 1000)
            .Select(i => (float)(i * 0.001))
            .ToArray();

        var profile = LayerScaleProfile.Compute(
            "test.layer",
            outputDim: 256,
            inputDim: 256,
            gradients,
            minWeightSeen: -1.0f,
            maxWeightSeen: 1.0f);

        Assert.True(profile.Epsilon > 0f);
        Assert.True(float.IsFinite(profile.Epsilon));
        Assert.True(profile.P99GradientMagnitude > 0f);
        Assert.True(profile.BucketCount > 0);
        Assert.True(profile.TernaryThreshold > 0);
    }

    [Fact]
    public void ModelScaleProfile_SerializationRoundTrips()
    {
        var profile = new ModelScaleProfile
        {
            ModelId = "test-model",
            ArchitectureHash = "ABCDEF0123456789",
            CalibratedAt = new DateTimeOffset(2026, 4, 14, 0, 0, 0, TimeSpan.Zero),
            CalibrationSteps = 500,
            Layers =
            [
                new LayerScaleProfile
                {
                    LayerName = "layer0.q",
                    OutputDimension = 256,
                    InputDimension = 256,
                    MaxGradientMagnitude = 0.1f,
                    P99GradientMagnitude = 0.05f,
                    MeanGradientMagnitude = 0.01f,
                    ObservedWeightRange = 2.0f,
                    MaxWeightMagnitude = 1.0f,
                    Epsilon = 1.5e-6f,
                    BucketCount = 20
                }
            ]
        };

        var path = Path.GetTempFileName();
        try
        {
            profile.SaveToFile(path);
            var loaded = ModelScaleProfile.LoadFromFile(path);

            Assert.Equal(profile.ModelId, loaded.ModelId);
            Assert.Equal(profile.ArchitectureHash, loaded.ArchitectureHash);
            Assert.Equal(profile.CalibrationSteps, loaded.CalibrationSteps);
            Assert.Single(loaded.Layers);
            Assert.Equal("layer0.q", loaded.Layers[0].LayerName);
            Assert.Equal(0.05f, loaded.Layers[0].P99GradientMagnitude);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void IntegerMasterWeight_UpdateStep_ChangesWeights()
    {
        var profile = CreateTestProfile();
        var layer = new IntegerMasterWeightLayer(profile);
        layer.InitializeFromTernary([1, 0, -1, 1]);

        var before = layer.ToFloat(0);
        layer.ApplyDelta(0, 0.001f);
        var after = layer.ToFloat(0);

        Assert.NotEqual(before, after);
    }

    [Fact]
    public void IntegerMasterWeight_CarryOnOverflow()
    {
        var profile = CreateTestProfile();
        var layer = new IntegerMasterWeightLayer(profile);
        layer.InitializeFromTernary([0, 0, 0, 0]);

        // Apply a large gradient that should trigger carry
        layer.ResetCarryCount();
        for (var i = 0; i < 100; i++)
        {
            layer.ApplyDelta(0, 1.0f); // Large gradient relative to epsilon
        }

        // After many large updates, carry should have happened
        Assert.True(layer.CarryCount > 0);
    }

    [Fact]
    public void IntegerMasterWeight_ProjectToTernary_MatchesThreshold()
    {
        var profile = CreateTestProfile();
        var layer = new IntegerMasterWeightLayer(profile);

        // Start from zero
        layer.InitializeFromTernary([0, 0, 0, 0]);

        // Apply enough positive gradient to push past ternary threshold
        for (var i = 0; i < 1000; i++)
        {
            layer.ApplyDelta(0, 0.01f);
        }

        var output = new sbyte[4];
        layer.ProjectToTernary(output);

        // After many positive updates, weight 0 should be positive
        Assert.Equal(1, output[0]);
        // Weight 1 untouched should be zero
        Assert.Equal(0, output[1]);
    }

    [Fact]
    public void GradientProfileCollector_ProducesFiniteScales()
    {
        var collector = new GradientProfileCollector();

        collector.RecordGradients("layer0", [0.01f, -0.02f, 0.015f, -0.005f]);
        collector.RecordWeights("layer0", [0.5f, -0.3f, 0.1f, -0.8f]);

        var config = new BitNetSharp.Core.Models.BitNetConfig();
        var profile = collector.BuildProfile("test", config, 100);

        Assert.Single(profile.Layers);
        Assert.True(float.IsFinite(profile.Layers[0].Epsilon));
        Assert.True(profile.Layers[0].Epsilon > 0f);
    }

    private static LayerScaleProfile CreateTestProfile() =>
        new()
        {
            LayerName = "test",
            OutputDimension = 2,
            InputDimension = 2,
            MaxGradientMagnitude = 0.1f,
            P99GradientMagnitude = 0.05f,
            MeanGradientMagnitude = 0.01f,
            ObservedWeightRange = 2.0f,
            MaxWeightMagnitude = 1.0f,
            Epsilon = 1e-5f,
            BucketCount = 10
        };
}
