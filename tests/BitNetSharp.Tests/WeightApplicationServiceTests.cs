#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Linq;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Extensions.Logging.Abstractions;
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

    [Fact]
    public void EnsureInitialized_creates_zero_vector_and_writes_initial_version()
    {
        var options = new CoordinatorOptions
        {
            InitialWeightDimension = 6,
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
        var svc = CreateService(new CoordinatorOptions { InitialWeightDimension = 4 });
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
            InitialWeightDimension = 4,
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
            InitialWeightDimension = 3,
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
        var svc = CreateService(new CoordinatorOptions { InitialWeightDimension = 4 });
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
            InitialWeightDimension = 2,
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
            InitialWeightDimension = 2,
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
        var svc = CreateService(new CoordinatorOptions { InitialWeightDimension = 2 });
        svc.EnsureInitialized();

        var result = svc.Apply(999, new float[] { 0f, 0f });
        Assert.False(result.Accepted);
        Assert.Contains("newer", result.Reason);
    }

    [Fact]
    public void EnsureInitialized_reloads_existing_version_from_disk()
    {
        // First instance seeds version 1 with a known vector.
        var first = CreateService(new CoordinatorOptions
        {
            InitialWeightDimension = 3,
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
            InitialWeightDimension = 3,
            BaseLearningRate = 0.1
        });
        second.EnsureInitialized();

        Assert.Equal(2, second.CurrentVersion);
        Assert.Equal(expected, second.Snapshot());
    }
}
#endif
