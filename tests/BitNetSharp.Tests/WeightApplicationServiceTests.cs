using System;
using System.IO;
using System.Linq;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Tests for the Phase D-4
/// <see cref="WeightApplicationService"/> that owns the in-memory
/// global weight vector, staleness compensation, and the persist-
/// to-disk side effect.
/// </summary>
public sealed class WeightApplicationServiceTests : IDisposable
{
    private readonly string _weightsDirectory;
    private readonly FileSystemWeightStore _store;

    public WeightApplicationServiceTests()
    {
        _weightsDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-weights-apply-{Guid.NewGuid():N}");
        _store = new FileSystemWeightStore(_weightsDirectory);
    }

    public void Dispose()
    {
        if (Directory.Exists(_weightsDirectory))
        {
            try { Directory.Delete(_weightsDirectory, recursive: true); } catch { }
        }
    }

    private WeightApplicationService CreateService(CoordinatorOptions options)
    {
        return new WeightApplicationService(
            _store,
            new StaticOptionsMonitor<CoordinatorOptions>(options),
            NullLogger<WeightApplicationService>.Instance);
    }

    private WeightApplicationService CreateServiceWithModelConfig(
        CoordinatorOptions options,
        ICoordinatorModelConfig modelConfig)
    {
        return new WeightApplicationService(
            _store,
            new StaticOptionsMonitor<CoordinatorOptions>(options),
            modelConfig,
            NullLogger<WeightApplicationService>.Instance);
    }

    private static ICoordinatorModelConfig BuildPresetConfig(string preset)
    {
        var opts = Options.Create(new CoordinatorOptions { ModelPreset = preset });
        return new CoordinatorModelConfig(opts);
    }

    [Fact]
    public void EnsureInitialized_creates_zero_vector_and_writes_initial_version()
    {
        var options = new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 6,
            InitialWeightVersion = 1
        };
        var svc = CreateService(options);

        svc.EnsureInitialized();

        Assert.Equal(1, svc.CurrentVersion);
        Assert.Equal(6, svc.Dimension);
        Assert.All(svc.Snapshot(), v => Assert.Equal(0f, v));
        Assert.NotNull(_store.TryGetManifest(1));
    }

    [Fact]
    public void EnsureInitialized_is_idempotent()
    {
        var svc = CreateService(new CoordinatorOptions { ModelPreset = "", InitialWeightDimension = 4 });
        svc.EnsureInitialized();
        var firstVersion = svc.CurrentVersion;
        svc.EnsureInitialized();
        Assert.Equal(firstVersion, svc.CurrentVersion);
    }

    [Fact]
    public void Apply_decrements_weights_by_lr_times_gradient_and_bumps_version()
    {
        var svc = CreateService(new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 4,
            BaseLearningRate = 0.1,
            StalenessAlpha = 0,
            MaxStalenessSteps = 10
        });
        svc.EnsureInitialized();

        var gradient = new float[] { 1f, 2f, 3f, 4f };
        var result = svc.Apply(baseVersion: 1, gradient);

        Assert.True(result.Accepted);
        Assert.Equal(2, result.NewVersion);
        Assert.Equal(0, result.Staleness);
        Assert.Equal(0.1f, result.EffectiveLearningRate, precision: 5);

        var after = svc.Snapshot();
        Assert.Equal(-0.1f, after[0], precision: 5);
        Assert.Equal(-0.2f, after[1], precision: 5);
        Assert.Equal(-0.3f, after[2], precision: 5);
        Assert.Equal(-0.4f, after[3], precision: 5);
    }

    [Fact]
    public void Apply_persists_new_version_blob_to_disk()
    {
        var svc = CreateService(new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 3,
            BaseLearningRate = 0.1,
            StalenessAlpha = 0
        });
        svc.EnsureInitialized();

        svc.Apply(1, new float[] { 1f, 0f, -1f });

        var manifest = _store.TryGetManifest(2);
        Assert.NotNull(manifest);
        using var stream = _store.TryOpenReadStream(2);
        using var memory = new MemoryStream();
        stream!.CopyTo(memory);
        var weights = WeightBlobCodec.Decode(memory.ToArray(), out var version);
        Assert.Equal(2, version);
        Assert.Equal(-0.1f, weights[0], precision: 5);
    }

    [Fact]
    public void Apply_rejects_gradient_with_wrong_shape()
    {
        var svc = CreateService(new CoordinatorOptions { ModelPreset = "", InitialWeightDimension = 4 });
        svc.EnsureInitialized();

        var result = svc.Apply(1, new float[] { 1f, 2f });

        Assert.False(result.Accepted);
        Assert.NotNull(result.Reason);
        Assert.Contains("does not match", result.Reason);
    }

    [Fact]
    public void Apply_scales_learning_rate_by_staleness()
    {
        var svc = CreateService(new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 2,
            BaseLearningRate = 0.1,
            StalenessAlpha = 0.5,
            MaxStalenessSteps = 5
        });
        svc.EnsureInitialized();

        // Apply a first gradient so the version moves to 2.
        svc.Apply(1, new float[] { 1f, 1f });

        // Second worker submits based on the old version 1, which is
        // now stale by 1 step. Effective lr = 0.1 / (1 + 1 * 0.5) = 0.0666...
        var result = svc.Apply(1, new float[] { 1f, 1f });

        Assert.True(result.Accepted);
        Assert.Equal(1, result.Staleness);
        Assert.InRange(result.EffectiveLearningRate, 0.066f, 0.067f);
    }

    [Fact]
    public void Apply_rejects_gradient_that_exceeds_max_staleness()
    {
        var svc = CreateService(new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 2,
            BaseLearningRate = 0.1,
            StalenessAlpha = 0,
            MaxStalenessSteps = 1
        });
        svc.EnsureInitialized();

        // Push version forward twice.
        svc.Apply(1, new float[] { 1f, 0f });
        svc.Apply(2, new float[] { 1f, 0f });

        // Now base_version = 1 is stale by 2 which exceeds max = 1.
        var result = svc.Apply(1, new float[] { 1f, 0f });
        Assert.False(result.Accepted);
        Assert.Contains("stale", result.Reason);
    }

    [Fact]
    public void Apply_rejects_gradient_from_future_base_version()
    {
        var svc = CreateService(new CoordinatorOptions { ModelPreset = "", InitialWeightDimension = 2 });
        svc.EnsureInitialized();

        var result = svc.Apply(999, new float[] { 0f, 0f });
        Assert.False(result.Accepted);
        Assert.Contains("newer", result.Reason);
    }

    // ── Track 7: model-preset initialization + migration ────────────

    [Fact]
    public void EnsureInitialized_with_model_config_sizes_vector_to_flat_length()
    {
        var modelConfig = BuildPresetConfig("small");
        var svc = CreateServiceWithModelConfig(
            new CoordinatorOptions { ModelPreset = "small" },
            modelConfig);

        svc.EnsureInitialized();

        // Small preset: 5174*256 + 4*(4*256*256 + 2*256*1024 + 1024*256) + 256*5174
        //             = 6,843,392 fp32 elements.
        Assert.Equal(6_843_392, modelConfig.FlatLength);
        Assert.Equal(modelConfig.FlatLength, svc.Dimension);
        Assert.Equal(1, svc.CurrentVersion);

        // The initial vector is not zeros: it is the FlatParameterPack
        // of a freshly-constructed BitNetTransformer, which uses
        // ParameterInitializer random weights, so at least one element
        // must be non-zero.
        var snapshot = svc.Snapshot();
        Assert.Contains(snapshot, v => v != 0f);
    }

    [Fact]
    public void Apply_with_model_config_round_trips_full_length_gradient()
    {
        var modelConfig = BuildPresetConfig("small");
        var svc = CreateServiceWithModelConfig(
            new CoordinatorOptions
            {
                ModelPreset = "small",
                BaseLearningRate = 0.01,
                StalenessAlpha = 0
            },
            modelConfig);
        svc.EnsureInitialized();

        var before = svc.Snapshot();
        var gradient = new float[modelConfig.FlatLength];
        gradient[0] = 1f;
        gradient[modelConfig.FlatLength - 1] = -2f;

        var result = svc.Apply(baseVersion: 1, gradient);

        Assert.True(result.Accepted);
        Assert.Equal(2, result.NewVersion);

        var after = svc.Snapshot();
        Assert.Equal(before[0] - 0.01f, after[0], precision: 5);
        Assert.Equal(before[^1] - 0.01f * -2f, after[^1], precision: 5);
    }

    [Fact]
    public void Apply_with_model_config_rejects_gradient_with_wrong_length()
    {
        var modelConfig = BuildPresetConfig("small");
        var svc = CreateServiceWithModelConfig(
            new CoordinatorOptions { ModelPreset = "small" },
            modelConfig);
        svc.EnsureInitialized();

        // A 4,096-element gradient (the legacy D-1 placeholder) must
        // be rejected when the coordinator is configured for the
        // real 6,843,392-element small preset.
        var legacyGradient = new float[4_096];
        var result = svc.Apply(1, legacyGradient);

        Assert.False(result.Accepted);
        Assert.NotNull(result.Reason);
        Assert.Contains("does not match", result.Reason);
    }

    [Fact]
    public void EnsureInitialized_resets_legacy_4096_persisted_weights_to_preset_length()
    {
        // Simulate the Phase D state: v1 on disk is a 4,096-element
        // zero vector written by the pre-Track-7 coordinator.
        var legacyBlob = WeightBlobCodec.Encode(1L, new float[4_096]);
        _store.SaveVersion(1L, legacyBlob);

        var modelConfig = BuildPresetConfig("small");
        var svc = CreateServiceWithModelConfig(
            new CoordinatorOptions { ModelPreset = "small" },
            modelConfig);

        svc.EnsureInitialized();

        // Service must NOT adopt the legacy 4096-element vector —
        // instead it logs a warning and initializes a fresh vector
        // at version 2 sized to the configured preset.
        Assert.Equal(2, svc.CurrentVersion);
        Assert.Equal(modelConfig.FlatLength, svc.Dimension);
        Assert.NotEqual(4_096, svc.Dimension);

        // Legacy v1 must stay on disk (immutable) and v2 must be
        // visible so workers can download it.
        Assert.NotNull(_store.TryGetManifest(1));
        Assert.NotNull(_store.TryGetManifest(2));
    }

    [Fact]
    public void CoordinatorModelConfig_exposes_expected_small_preset_values()
    {
        var modelConfig = BuildPresetConfig("small");

        Assert.Equal("small", modelConfig.PresetName);
        Assert.Equal(6_843_392, modelConfig.FlatLength);
        Assert.Equal(FlatParameterPack.ComputeLength(modelConfig.Config), modelConfig.FlatLength);
        Assert.Contains("preset=small", modelConfig.ToDisplayString());
    }

    [Fact]
    public void EnsureInitialized_reloads_existing_version_from_disk()
    {
        // First instance seeds version 1 with a known vector.
        var first = CreateService(new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 3,
            BaseLearningRate = 0.1
        });
        first.EnsureInitialized();
        first.Apply(1, new float[] { 1f, 2f, 3f });
        var expected = first.Snapshot();
        Assert.Equal(2, first.CurrentVersion);

        // Second instance shares the same store and should pick up
        // version 2 on EnsureInitialized rather than starting fresh.
        var second = CreateService(new CoordinatorOptions
        {
            ModelPreset = "", InitialWeightDimension = 3,
            BaseLearningRate = 0.1
        });
        second.EnsureInitialized();

        Assert.Equal(2, second.CurrentVersion);
        Assert.Equal(expected, second.Snapshot());
    }
}
