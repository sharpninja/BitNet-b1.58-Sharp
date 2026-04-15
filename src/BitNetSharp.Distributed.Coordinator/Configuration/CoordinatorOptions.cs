using System;
using System.Collections.Generic;

namespace BitNetSharp.Distributed.Coordinator.Configuration;

/// <summary>
/// Strongly-typed configuration binding for the coordinator host. Bound
/// from the <c>Coordinator:</c> section of <c>appsettings.json</c> or
/// from environment variables at startup.
///
/// <para>
/// Environment-variable convention: ASP.NET Core treats <c>__</c> as
/// the hierarchy separator, so a client list entry looks like
/// <c>Coordinator__WorkerClients__0__ClientId=worker-alpha</c>.
/// </para>
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
    /// Lifetime in seconds of the JWT access tokens issued to workers
    /// by the identity server. Kept short so natural expiry bounds the
    /// damage if a token is leaked; immediate invalidation is handled
    /// by the revocation registry.
    /// </summary>
    public int AccessTokenLifetimeSeconds { get; set; } = 3600;

    /// <summary>
    /// Public base URL at which this coordinator is reachable. The
    /// identity server and the admin OIDC client both use this value
    /// as the OpenID Connect issuer / authority so self-referential
    /// discovery works. In production this is the ngrok reserved
    /// domain; in local development it is whatever HTTPS URL Kestrel
    /// binds to.
    /// </summary>
    public string BaseUrl { get; set; } = "https://localhost:5001";

    /// <summary>
    /// Dimension of the global weight vector the coordinator tracks
    /// in memory. Phase D-4 uses a flat fp32 vector of this size.
    /// Defaults to 4096 which is small enough to fit comfortably in
    /// the test harness and large enough to exercise real int8
    /// quantization behavior.
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
    /// List of OAuth 2.0 client-credentials clients that are allowed
    /// to authenticate as workers. Populated from environment at
    /// startup (see class remarks for env-var naming). Add one entry
    /// per worker machine.
    /// </summary>
    public List<WorkerClientOptions> WorkerClients { get; set; } = new();

    /// <summary>
    /// Credentials the <c>/admin/*</c> endpoints check with HTTP Basic
    /// auth. The admin pages display API keys in plain text so you can
    /// copy them to worker machines; protect them accordingly.
    /// </summary>
    public AdminOptions Admin { get; set; } = new();
}

/// <summary>
/// Configuration entry for a single OAuth 2.0 client that is allowed to
/// authenticate as a worker via machine-to-machine login.
/// </summary>
public sealed class WorkerClientOptions
{
    /// <summary>
    /// OAuth client_id. Doubles as the persistent worker identity in
    /// the coordinator's <c>workers</c> table.
    /// </summary>
    public string ClientId { get; set; } = string.Empty;

    /// <summary>
    /// OAuth client_secret. Treated like an API key by the operator
    /// (displayed on the admin page, copied to worker env vars).
    /// </summary>
    public string ClientSecret { get; set; } = string.Empty;

    /// <summary>
    /// Human-friendly label for the admin page.
    /// </summary>
    public string DisplayName { get; set; } = string.Empty;
}

/// <summary>
/// Credentials for the coordinator's HTTP Basic-auth-protected admin
/// area. Both username and password are read from environment so no
/// secrets live in source control.
/// </summary>
public sealed class AdminOptions
{
    public string Username { get; set; } = "admin";
    public string Password { get; set; } = string.Empty;
    public string AdminScheme { get; set; } = "BitNetCoordinator";
}
