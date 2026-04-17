using System;

namespace BitNetSharp.Distributed.Coordinator.Configuration;

/// <summary>
/// Strongly-typed configuration binding for the coordinator host. Bound
/// from the <c>Coordinator:</c> section of <c>appsettings.json</c> or
/// from environment variables at startup.
/// </summary>
public sealed class CoordinatorOptions
{
    public const string SectionName = "Coordinator";

    /// <summary>
    /// Path to the coordinator SQLite database file. Relative paths
    /// resolve against the process working directory.
    /// </summary>
    public string DatabasePath { get; set; } = "coordinator.db";

    /// <summary>
    /// How often the coordinator expects each worker to send a
    /// heartbeat. Also the value echoed back in the registration
    /// response so workers can sync their cadence.
    /// </summary>
    public int HeartbeatIntervalSeconds { get; set; } = 30;

    /// <summary>
    /// Threshold past which an <c>Active</c> worker with no recent
    /// heartbeat is swept to <c>Gone</c>. Should be several multiples
    /// of <see cref="HeartbeatIntervalSeconds"/> to absorb transient
    /// network hiccups.
    /// </summary>
    public int StaleWorkerThresholdSeconds { get; set; } = 300;

    /// <summary>
    /// Starting weight version handed to a freshly registered worker.
    /// Kept as configuration so bootstrap-from-checkpoint workflows
    /// can start the pool at a non-zero version.
    /// </summary>
    public long InitialWeightVersion { get; set; } = 1;

    /// <summary>
    /// Target wall-clock duration of a single task in seconds.
    /// Defaults to ten minutes.
    /// </summary>
    public int TargetTaskDurationSeconds { get; set; } = 600;

    /// <summary>
    /// Ratio of calibration throughput to full-training-step throughput.
    /// See <see cref="BitNetSharp.Distributed.Contracts.TaskSizingCalculator"/>
    /// for the meaning. Clamped to [0.01, 1.0] inside the calculator.
    /// </summary>
    public double FullStepEfficiency { get; set; } = 0.25d;

    /// <summary>
    /// Public base URL at which this coordinator is reachable. The
    /// identity server (admin OIDC only) uses this value as the
    /// OpenID Connect issuer / authority so self-referential
    /// discovery works.
    /// </summary>
    public string BaseUrl { get; set; } = "https://localhost:5001";

    /// <summary>
    /// Named model preset that determines the global weight vector
    /// dimension. Recognized: "small" (~7M), "medium" (~56M),
    /// "large" (~121M). When set, overrides
    /// <see cref="InitialWeightDimension"/> with the preset's
    /// TotalWeightElements.
    /// </summary>
    public string ModelPreset { get; set; } = "small";

    /// <summary>
    /// Dimension of the global weight vector the coordinator tracks
    /// in memory. Overridden by <see cref="ModelPreset"/> when set.
    /// Kept as fallback for custom non-preset configs.
    /// </summary>
    public int InitialWeightDimension { get; set; } = 4096;

    /// <summary>
    /// Base learning rate applied to decoded worker gradients. The
    /// effective rate used during
    /// <see cref="BitNetSharp.Distributed.Coordinator.Services.WeightApplicationService.Apply"/>
    /// is scaled down for stale gradients via
    /// <see cref="StalenessAlpha"/>.
    /// </summary>
    public double BaseLearningRate { get; set; } = 0.01d;

    /// <summary>
    /// Linear staleness penalty: effective_lr =
    /// base_lr / (1 + staleness * alpha). Higher alpha means the
    /// coordinator trusts stale gradients less. Zero disables
    /// staleness compensation entirely.
    /// </summary>
    public double StalenessAlpha { get; set; } = 0.5d;

    /// <summary>
    /// Maximum number of weight versions a gradient is allowed to
    /// lag behind the current global version before the coordinator
    /// rejects the submission outright. Ten is the v1 default — see
    /// the distributed training design notes.
    /// </summary>
    public long MaxStalenessSteps { get; set; } = 10;

    /// <summary>
    /// Number of days to retain <c>gradient_events</c> telemetry
    /// rows before the hourly prune service deletes them.
    /// </summary>
    public int TelemetryRetentionDays { get; set; } = 7;

    /// <summary>
    /// Number of days to retain <c>worker_logs</c> rows before the
    /// hourly prune service deletes them.
    /// </summary>
    public int LogRetentionDays { get; set; } = 3;

    /// <summary>
    /// Directory where nightly backup snapshots are written. Empty
    /// value resolves to <c>{DatabasePath-directory}/backups</c>.
    /// Each iteration writes a <c>yyyyMMdd'T'HHmmss</c> subdir holding
    /// the SQLite copy + weight blobs.
    /// </summary>
    public string BackupRoot { get; set; } = string.Empty;

    /// <summary>
    /// Number of days a backup subdir is retained before
    /// <see cref="BitNetSharp.Distributed.Coordinator.Services.DatabaseBackupService"/>
    /// prunes it on the next iteration.
    /// </summary>
    public int BackupRetentionDays { get; set; } = 14;

    /// <summary>
    /// Wall-clock interval between backup iterations. Clamped to a
    /// minimum of 1 hour inside the service.
    /// </summary>
    public int BackupIntervalHours { get; set; } = 24;

    /// <summary>
    /// Single shared API key every worker must present in the
    /// <c>X-Api-Key</c> header to hit the worker endpoints. Set via
    /// environment variable <c>Coordinator__WorkerApiKey</c> by the
    /// operator. Rotating the key = edit the env var, restart the
    /// coordinator; every worker with the old key is instantly
    /// locked out.
    /// </summary>
    public string WorkerApiKey { get; set; } = string.Empty;

    /// <summary>
    /// Credentials the admin dashboard's OIDC-backed login form
    /// validates against. The admin pages are separate from the
    /// worker plane; workers authenticate with the shared API key
    /// instead.
    /// </summary>
    public AdminOptions Admin { get; set; } = new();

    /// <summary>
    /// Anthropic API key used by
    /// <see cref="BitNetSharp.Distributed.Coordinator.Services.SonnetAsrCorpusGenerator"/>
    /// to query Claude Sonnet for synthetic ASR-noisy transcripts. Set
    /// via env var <c>Coordinator__AnthropicApiKey</c>. Empty value
    /// aborts the generator before any HTTP call.
    /// </summary>
    public string AnthropicApiKey { get; set; } = string.Empty;

    /// <summary>
    /// Anthropic model id passed in the Messages API request body.
    /// Defaults to the current Sonnet release.
    /// </summary>
    public string AnthropicModel { get; set; } = "claude-sonnet-4-6";

    /// <summary>
    /// Max concurrent in-flight Messages-API calls. Matches the
    /// Anthropic tier-1 rate-limit headroom; raise cautiously.
    /// </summary>
    public int AsrMaxConcurrency { get; set; } = 4;

    /// <summary>
    /// Cumulative USD cost ceiling per generator run. Enforced after
    /// each batch using the <c>usage</c> block the Messages API
    /// returns. Once the running total exceeds this, the generator
    /// stops and returns a partial manifest.
    /// </summary>
    public decimal AsrCostCapUsd { get; set; } = 5.00m;

    /// <summary>
    /// Shard-id prefix for generated ASR shards. Workers select ASR
    /// shards via this prefix so they stay distinct from the
    /// deterministic <c>truckmate-v2-</c> corpus.
    /// </summary>
    public string AsrShardPrefix { get; set; } = "asr-v1-";
}

/// <summary>
/// Credentials for the coordinator's admin dashboard login form.
/// Both username and password are read from environment so no secrets
/// live in source control.
/// </summary>
public sealed class AdminOptions
{
    public string Username { get; set; } = "admin";
    public string Password { get; set; } = string.Empty;
    public string AdminScheme { get; set; } = "BitNetCoordinator";
}
