using BitNetSharp.Distributed.Contracts;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Locks the Truck Mate model presets so accidental config changes
/// don't silently alter the architecture the fleet trains.
/// </summary>
public sealed class TruckMateModelPresetsTests
{
    [Fact]
    public void Small_preset_is_approximately_7M_params()
    {
        var preset = TruckMateModelPresets.Small();
        Assert.Equal("truckmate-small", preset.Name);
        Assert.Equal(256, preset.Dimension);
        Assert.Equal(4, preset.LayerCount);
        Assert.InRange(preset.EstimatedParameters, 5_000_000L, 10_000_000L);
    }

    [Fact]
    public void Medium_preset_is_approximately_56M_params()
    {
        var preset = TruckMateModelPresets.Medium();
        Assert.Equal("truckmate-medium", preset.Name);
        Assert.Equal(512, preset.Dimension);
        Assert.Equal(12, preset.LayerCount);
        Assert.InRange(preset.EstimatedParameters, 50_000_000L, 60_000_000L);
    }

    [Fact]
    public void Large_preset_is_approximately_121M_params()
    {
        var preset = TruckMateModelPresets.Large();
        Assert.Equal("truckmate-large", preset.Name);
        Assert.Equal(768, preset.Dimension);
        Assert.Equal(12, preset.LayerCount);
        Assert.InRange(preset.EstimatedParameters, 115_000_000L, 130_000_000L);
    }

    [Fact]
    public void GetPreset_resolves_by_name_case_insensitive()
    {
        Assert.Equal("truckmate-small",  TruckMateModelPresets.GetPreset("Small").Name);
        Assert.Equal("truckmate-medium", TruckMateModelPresets.GetPreset("MEDIUM").Name);
        Assert.Equal("truckmate-large",  TruckMateModelPresets.GetPreset("large").Name);
        Assert.Equal("truckmate-medium", TruckMateModelPresets.GetPreset("unknown").Name); // default
    }

    [Fact]
    public void VocabSize_override_propagates()
    {
        var preset = TruckMateModelPresets.GetPreset("small", vocabSizeOverride: 10_000);
        Assert.Equal(10_000, preset.VocabSize);
    }

    [Fact]
    public void ToDisplayString_includes_key_numbers()
    {
        var display = TruckMateModelPresets.Medium().ToDisplayString();
        Assert.Contains("truckmate-medium", display);
        Assert.Contains("512", display);
        Assert.Contains("12", display);
        Assert.Contains("M params", display);
    }

    [Fact]
    public void All_presets_satisfy_BitNetConfig_invariants()
    {
        foreach (var name in new[] { "small", "medium", "large" })
        {
            var p = TruckMateModelPresets.GetPreset(name);
            // Dimension divisible by HeadCount
            Assert.Equal(0, p.Dimension % p.HeadCount);
            // HeadDimension is even (for rotary embeddings)
            Assert.Equal(0, (p.Dimension / p.HeadCount) % 2);
            // Positive everything
            Assert.True(p.VocabSize > 0);
            Assert.True(p.Dimension > 0);
            Assert.True(p.HiddenDimension > 0);
            Assert.True(p.LayerCount > 0);
            Assert.True(p.MaxSequenceLength > 0);
        }
    }
}
