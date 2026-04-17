using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Worker;
using Xunit.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests for <see cref="RealTrainingGradient"/>. Verifies that
/// the worker can consume a flat parameter vector, train locally on
/// a tiny synthetic shard, and return a finite non-zero delta of
/// matching length.
/// </summary>
public sealed class RealTrainingGradientTests
{
    private readonly ITestOutputHelper _output;

    public RealTrainingGradientTests(ITestOutputHelper output)
    {
        _output = output;
    }
    private static BitNetConfig TinyConfig() => new(
        vocabSize: 32,
        dimension: 16,
        hiddenDimension: 32,
        layerCount: 2,
        headCount: 2,
        maxSequenceLength: 32);

    [Fact]
    public void ComputeGradient_returns_delta_of_matching_length_with_finite_values()
    {
        var cfg = TinyConfig();
        var transformer = new BitNetTransformer(cfg, seed: 42);
        var currentFlat = FlatParameterPack.Pack(transformer);

        // Single 32-token synthetic sequence, deterministic.
        var rng = new Random(7);
        var sequence = new int[32];
        for (var i = 0; i < sequence.Length; i++)
        {
            sequence[i] = rng.Next(0, cfg.VocabSize);
        }

        var delta = RealTrainingGradient.ComputeGradient(
            currentFlat,
            new[] { sequence },
            cfg,
            localSteps: 1);

        Assert.Equal(currentFlat.Length, delta.Length);

        double norm = 0d;
        for (var i = 0; i < delta.Length; i++)
        {
            Assert.False(float.IsNaN(delta[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(delta[i]), $"Inf at index {i}");
            norm += delta[i] * (double)delta[i];
        }
        norm = Math.Sqrt(norm);

        Assert.True(norm > 0d, $"Expected non-zero gradient norm after 1 training step, got {norm}");

        _output.WriteLine($"Tiny-model 1-step gradient: length={delta.Length}, L2 norm={norm:F6}");
    }

    [Fact]
    public void Diagnostic_small_preset_compute_length()
    {
        var preset = TruckMateModelPresets.GetPreset("small");
        var cfg = new BitNetConfig(
            vocabSize: preset.VocabSize,
            dimension: preset.Dimension,
            hiddenDimension: preset.HiddenDimension,
            layerCount: preset.LayerCount,
            headCount: preset.HeadCount,
            maxSequenceLength: preset.MaxSequenceLength);
        var len = FlatParameterPack.ComputeLength(cfg);

        _output.WriteLine($"Small preset FlatParameterPack.ComputeLength = {len}");
        _output.WriteLine($"  vocab={cfg.VocabSize}, dim={cfg.Dimension}, hidden={cfg.HiddenDimension}, layers={cfg.LayerCount}, seq={cfg.MaxSequenceLength}");
        Assert.True(len > 0);
    }

    [Fact]
    public void ComputeGradient_rejects_flat_vector_of_wrong_length()
    {
        var cfg = TinyConfig();
        var wrong = new float[FlatParameterPack.ComputeLength(cfg) + 10];

        var ex = Assert.Throws<ArgumentException>(() =>
            RealTrainingGradient.ComputeGradient(
                wrong,
                new[] { new[] { 1, 2, 3 } },
                cfg,
                localSteps: 1));

        Assert.Contains("does not match expected", ex.Message);
    }

    [Fact]
    public void ComputeGradient_returns_zero_delta_when_shard_empty()
    {
        var cfg = TinyConfig();
        var transformer = new BitNetTransformer(cfg, seed: 5);
        var currentFlat = FlatParameterPack.Pack(transformer);

        var delta = RealTrainingGradient.ComputeGradient(
            currentFlat,
            Array.Empty<int[]>(),
            cfg,
            localSteps: 1);

        Assert.Equal(currentFlat.Length, delta.Length);
        for (var i = 0; i < delta.Length; i++)
        {
            Assert.Equal(0f, delta[i]);
        }
    }
}
