using System;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
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
/// Phase A Track 7: the in-memory vector is the full BitNet flat
/// parameter pack sized by <see cref="FlatParameterPack.ComputeLength"/>
/// against the configured <see cref="ICoordinatorModelConfig"/>. For
/// preset "small" that is ~6.84M fp32 elements = ~26.1 MiB on-wire.
/// Fresh initialization packs a <see cref="BitNetTransformer"/> so
/// workers downloading version 1 see the same random starting point
/// they would materialize locally. Persisted weights whose length does
/// not match the configured preset are treated as a fresh training
/// run: a WARNING is logged and the service re-initializes — this is
/// the migration path for the legacy 4096-element placeholder shipped
/// in Phase D-1 through D-4.
/// </para>
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
/// Aggregation policy (Track 7 scope): plain additive SGD —
/// <c>w -= effective_lr * gradient</c>. Mean-across-workers and
/// variance-rejection aggregation is a documented follow-up for
/// Track 9.
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
    private readonly ICoordinatorModelConfig? _modelConfig;
    private readonly ILogger<WeightApplicationService> _logger;
    private readonly object _gate = new();

    private float[] _current = Array.Empty<float>();
    private long _currentVersion;
    private bool _initialized;

    /// <summary>
    /// Full-feature constructor used by the coordinator host: takes
    /// the resolved <see cref="ICoordinatorModelConfig"/> so the
    /// service can size the in-memory vector to match the configured
    /// preset and seed the initial version from a
    /// <see cref="BitNetTransformer"/>.
    /// </summary>
    public WeightApplicationService(
        FileSystemWeightStore store,
        IOptionsMonitor<CoordinatorOptions> options,
        ICoordinatorModelConfig modelConfig,
        ILogger<WeightApplicationService> logger)
    {
        _store = store;
        _options = options;
        _modelConfig = modelConfig;
        _logger = logger;
    }

    /// <summary>
    /// Legacy constructor kept so the existing unit tests (which use
    /// the small <c>InitialWeightDimension</c> knob to drive the
    /// service with tiny synthetic vectors) continue to compile and
    /// pass. When <paramref name="modelConfig"/> is null, the service
    /// falls back to the old behaviour: use the configured preset's
    /// <c>TotalWeightElements</c> when non-empty, otherwise the raw
    /// <see cref="CoordinatorOptions.InitialWeightDimension"/> knob.
    /// </summary>
    public WeightApplicationService(
        FileSystemWeightStore store,
        IOptionsMonitor<CoordinatorOptions> options,
        ILogger<WeightApplicationService> logger)
        : this(store, options, modelConfig: null!, logger)
    {
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
    /// or initializes a fresh vector sized to the configured preset's
    /// <see cref="FlatParameterPack.ComputeLength"/>. Idempotent —
    /// second call is a no-op.
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
            var expectedLength = ResolveExpectedLength(opts);

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
                        if (_modelConfig is not null && storedWeights.Length != expectedLength)
                        {
                            _logger.LogWarning(
                                "Persisted weight version {Version} has {Got} elements but configured preset '{Preset}' expects {Expected}. "
                                + "Treating as a fresh training run — the legacy vector will be retained on disk but a new v{NewVersion} will be written.",
                                storedVersion,
                                storedWeights.Length,
                                _modelConfig.PresetName,
                                expectedLength,
                                storedVersion + 1);
                            InitializeFromPreset(opts, expectedLength, startVersion: storedVersion + 1);
                            return;
                        }

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

            InitializeFromPreset(opts, expectedLength, startVersion: Math.Max(1L, opts.InitialWeightVersion));
        }
    }

    private int ResolveExpectedLength(CoordinatorOptions opts)
    {
        // Prefer the fully-wired model config — it uses
        // FlatParameterPack.ComputeLength which is the canonical length
        // both workers and coordinator agree on.
        if (_modelConfig is not null)
        {
            return _modelConfig.FlatLength;
        }

        // Fallback for the legacy two-arg constructor used by a handful
        // of existing unit tests. Preserves pre-Track-7 behaviour so
        // those tests keep passing.
        if (!string.IsNullOrWhiteSpace(opts.ModelPreset))
        {
            var preset = TruckMateModelPresets.GetPreset(opts.ModelPreset);
            return (int)Math.Min(preset.TotalWeightElements, int.MaxValue);
        }

        return Math.Max(1, opts.InitialWeightDimension);
    }

    private void InitializeFromPreset(CoordinatorOptions opts, int expectedLength, long startVersion)
    {
        float[] initial;
        if (_modelConfig is not null)
        {
            // Seed from a fresh BitNetTransformer so every worker
            // downloading version N sees the same starting point the
            // first worker would materialize locally. This also
            // guarantees the flat layout matches FlatParameterPack.Pack
            // byte-for-byte (token embeddings + BitLinear masters in
            // canonical order).
            var transformer = new BitNetTransformer(_modelConfig.Config);
            initial = FlatParameterPack.Pack(transformer);
            if (initial.Length != expectedLength)
            {
                throw new InvalidOperationException(
                    $"FlatParameterPack.Pack produced {initial.Length} elements but ComputeLength said {expectedLength}.");
            }
        }
        else
        {
            // Legacy zero-init path — preserved for the pre-Track-7
            // unit-test harness.
            initial = new float[Math.Max(1, expectedLength)];
        }

        _current = initial;
        _currentVersion = startVersion;
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
            "Initialized fresh weight vector at version {Version} with {Dimension} elements{Origin}.",
            _currentVersion,
            _current.Length,
            _modelConfig is null ? "" : $" (preset '{_modelConfig.PresetName}')");
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

            // Plain additive SGD — w -= lr * g. Per-worker aggregation
            // (mean / variance-reject) is a Track 9 follow-up.
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
