using System;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Thread-safe singleton that owns the coordinator's in-memory copy
/// of the global fp32 weight vector and drives the decode → apply →
/// persist → bump-version flow that
/// <see cref="Cqrs.Commands.SubmitGradientCommandHandler"/> calls
/// from every successful <c>/gradient</c> submission.
///
/// <para>
/// Staleness compensation: every submission carries the
/// <c>base_weight_version</c> the worker trained against. When that
/// version is older than the current global version the service
/// computes <c>staleness = current - base</c>, rejects the
/// submission if it exceeds <see cref="CoordinatorOptions.MaxStalenessSteps"/>,
/// and otherwise scales the learning rate by
/// <c>1 / (1 + staleness * alpha)</c>. This is the classic async
/// SGD staleness correction.
/// </para>
///
/// <para>
/// Every successful apply writes the new weight blob to
/// <see cref="FileSystemWeightStore"/> as an immutable version so
/// workers can pull it from <c>/weights/{version}</c> before their
/// next task. The mutation itself is guarded by a lock so concurrent
/// submissions serialize cleanly.
/// </para>
/// </summary>
public sealed class WeightApplicationService
{
    private readonly FileSystemWeightStore _store;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;
    private readonly ILogger<WeightApplicationService> _logger;
    private readonly object _gate = new();

    private float[] _current = Array.Empty<float>();
    private long _currentVersion;
    private bool _initialized;

    public WeightApplicationService(
        FileSystemWeightStore store,
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<WeightApplicationService> logger)
    {
        _store = store;
        _options = options;
        _logger = logger;
    }

    /// <summary>Version number of the weights currently in memory.</summary>
    public long CurrentVersion
    {
        get
        {
            lock (_gate)
            {
                return _currentVersion;
            }
        }
    }

    /// <summary>Dimension of the weight vector.</summary>
    public int Dimension
    {
        get
        {
            lock (_gate)
            {
                return _current.Length;
            }
        }
    }

    /// <summary>
    /// Loads the highest-numbered weight version from disk (if any)
    /// or initializes a fresh zero vector at
    /// <see cref="CoordinatorOptions.InitialWeightVersion"/>.
    /// Idempotent — second call is a no-op.
    /// </summary>
    public void EnsureInitialized()
    {
        lock (_gate)
        {
            if (_initialized)
            {
                return;
            }

            var opts = _options.CurrentValue;
            var latest = _store.GetLatestVersion();
            if (latest is long existing)
            {
                using var stream = _store.TryOpenReadStream(existing);
                if (stream is not null)
                {
                    using var ms = new System.IO.MemoryStream();
                    stream.CopyTo(ms);
                    var bytes = ms.ToArray();
                    if (WeightBlobCodec.TryDecode(bytes, out var storedVersion, out var storedWeights, out var error))
                    {
                        _current = storedWeights;
                        _currentVersion = storedVersion;
                        _initialized = true;
                        _logger.LogInformation(
                            "Loaded weight version {Version} with {Dimension} elements from disk.",
                            _currentVersion,
                            _current.Length);
                        return;
                    }

                    _logger.LogWarning("Weight version {Version} on disk is malformed: {Error}. Re-initializing.", existing, error);
                }
            }

            // No weights on disk yet — materialize a zero vector at the
            // configured initial version + dimension and persist it.
            // If a model preset is configured, use its total parameter
            // count as the weight vector dimension instead of the raw
            // InitialWeightDimension knob.
            var dimension = opts.InitialWeightDimension;
            if (!string.IsNullOrWhiteSpace(opts.ModelPreset))
            {
                var preset = BitNetSharp.Distributed.Contracts.TruckMateModelPresets.GetPreset(opts.ModelPreset);
                dimension = (int)Math.Min(preset.TotalWeightElements, int.MaxValue);
                _logger.LogInformation(
                    "Using model preset {Preset}: {Display}",
                    opts.ModelPreset,
                    preset.ToDisplayString());
            }

            _current = new float[Math.Max(1, dimension)];
            _currentVersion = Math.Max(1L, opts.InitialWeightVersion);
            var blob = WeightBlobCodec.Encode(_currentVersion, _current);
            try
            {
                _store.SaveVersion(_currentVersion, blob);
            }
            catch (InvalidOperationException)
            {
                // Already exists (e.g. from a prior run that crashed
                // before it could decode). That's fine; just log.
                _logger.LogInformation(
                    "Weight version {Version} already present on disk; skipping initial save.",
                    _currentVersion);
            }

            _initialized = true;
            _logger.LogInformation(
                "Initialized fresh weight vector at version {Version} with {Dimension} zero elements.",
                _currentVersion,
                _current.Length);
        }
    }

    /// <summary>
    /// Applies a decoded gradient against the global weight vector
    /// and persists the new version to disk. Returns the outcome of
    /// the apply, including the new version, the effective learning
    /// rate that was used, and a reject reason when the submission
    /// was dropped (stale or wrong shape).
    /// </summary>
    public WeightApplicationResult Apply(long baseVersion, float[] gradient)
    {
        ArgumentNullException.ThrowIfNull(gradient);
        EnsureInitialized();

        var opts = _options.CurrentValue;
        var baseLearningRate = opts.BaseLearningRate;
        var alpha = opts.StalenessAlpha;
        var maxStaleness = opts.MaxStalenessSteps;

        lock (_gate)
        {
            if (gradient.Length != _current.Length)
            {
                return WeightApplicationResult.Rejected(
                    reason: $"Gradient length {gradient.Length} does not match weight dimension {_current.Length}.",
                    staleness: 0,
                    currentVersion: _currentVersion);
            }

            var staleness = _currentVersion - baseVersion;
            if (staleness < 0)
            {
                return WeightApplicationResult.Rejected(
                    reason: $"Base weight version {baseVersion} is newer than current {_currentVersion}.",
                    staleness: 0,
                    currentVersion: _currentVersion);
            }

            if (staleness > maxStaleness)
            {
                return WeightApplicationResult.Rejected(
                    reason: $"Gradient is stale by {staleness} versions (max {maxStaleness}).",
                    staleness: staleness,
                    currentVersion: _currentVersion);
            }

            var effectiveLr = (float)(baseLearningRate / (1d + Math.Max(0L, staleness) * Math.Max(0d, alpha)));

            for (var i = 0; i < _current.Length; i++)
            {
                _current[i] -= effectiveLr * gradient[i];
            }

            var newVersion = _currentVersion + 1;
            var blob = WeightBlobCodec.Encode(newVersion, _current);
            _store.SaveVersion(newVersion, blob);
            _currentVersion = newVersion;

            _logger.LogDebug(
                "Applied gradient against base version {BaseVersion} (staleness {Staleness}, lr {Lr:F4}) to produce version {NewVersion}.",
                baseVersion,
                staleness,
                effectiveLr,
                newVersion);

            return WeightApplicationResult.Applied(
                newVersion: newVersion,
                staleness: staleness,
                effectiveLearningRate: effectiveLr);
        }
    }

    /// <summary>
    /// Copies the current weight vector out to the caller so tests
    /// can assert on the post-apply state without touching internals.
    /// </summary>
    public float[] Snapshot()
    {
        lock (_gate)
        {
            var copy = new float[_current.Length];
            Array.Copy(_current, copy, _current.Length);
            return copy;
        }
    }
}

/// <summary>
/// Discriminated result type returned by
/// <see cref="WeightApplicationService.Apply"/>. <c>Accepted=true</c>
/// means the gradient was applied and <see cref="NewVersion"/> is the
/// version to hand the worker so it can refresh on its next task;
/// <c>Accepted=false</c> means the submission was rejected for the
/// given <see cref="Reason"/>.
/// </summary>
public sealed record WeightApplicationResult(
    bool Accepted,
    long NewVersion,
    long Staleness,
    float EffectiveLearningRate,
    string? Reason)
{
    public static WeightApplicationResult Applied(long newVersion, long staleness, float effectiveLearningRate) =>
        new(Accepted: true, NewVersion: newVersion, Staleness: staleness, EffectiveLearningRate: effectiveLearningRate, Reason: null);

    public static WeightApplicationResult Rejected(string reason, long staleness, long currentVersion) =>
        new(Accepted: false, NewVersion: currentVersion, Staleness: staleness, EffectiveLearningRate: 0f, Reason: reason);
}
