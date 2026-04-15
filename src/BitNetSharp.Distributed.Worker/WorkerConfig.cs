using System;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Strongly-typed view of the environment configuration read by a
/// distributed training worker. All values come from environment
/// variables so the same binary can run identically in a container,
/// on bare metal, or inside CI.
///
/// <para>
/// The worker authenticates to the coordinator using OAuth 2.0
/// client credentials (Duende IdentityServer's machine-login flow),
/// so each worker needs a per-machine <see cref="ClientId"/> /
/// <see cref="ClientSecret"/> pair provisioned by the operator on
/// the coordinator's admin page.
/// </para>
/// </summary>
internal sealed record WorkerConfig(
    Uri CoordinatorUrl,
    string ClientId,
    string ClientSecret,
    string WorkerName,
    int CpuThreads,
    TimeSpan HeartbeatInterval,
    TimeSpan ShutdownGrace,
    string HealthBeaconPath,
    string LogLevel)
{
    public const string EnvCoordinatorUrl   = "BITNET_COORDINATOR_URL";
    public const string EnvClientId         = "BITNET_CLIENT_ID";
    public const string EnvClientSecret     = "BITNET_CLIENT_SECRET";
    public const string EnvWorkerName       = "BITNET_WORKER_NAME";
    public const string EnvCpuThreads       = "BITNET_CPU_THREADS";
    public const string EnvHeartbeatSeconds = "BITNET_HEARTBEAT_SECONDS";
    public const string EnvShutdownSeconds  = "BITNET_SHUTDOWN_SECONDS";
    public const string EnvHealthBeaconPath = "BITNET_HEALTH_BEACON";
    public const string EnvLogLevel         = "BITNET_LOG_LEVEL";

    /// <summary>
    /// Reads configuration from the process environment. Fails fast
    /// (by throwing <see cref="InvalidOperationException"/>) if any
    /// required value is missing or malformed so Docker health checks
    /// surface the mistake in logs instead of looping silently.
    /// </summary>
    public static WorkerConfig FromEnvironment()
    {
        var coordinatorRaw = RequireEnv(EnvCoordinatorUrl);
        if (!Uri.TryCreate(coordinatorRaw, UriKind.Absolute, out var coordinatorUrl)
            || (coordinatorUrl.Scheme != Uri.UriSchemeHttp && coordinatorUrl.Scheme != Uri.UriSchemeHttps))
        {
            throw new InvalidOperationException(
                $"{EnvCoordinatorUrl} must be an absolute http(s) URL. Got '{coordinatorRaw}'.");
        }

        var clientId     = RequireEnv(EnvClientId);
        var clientSecret = RequireEnv(EnvClientSecret);

        var workerName = Environment.GetEnvironmentVariable(EnvWorkerName);
        if (string.IsNullOrWhiteSpace(workerName))
        {
            workerName = Environment.MachineName;
        }

        var cpuThreads = ParsePositiveInt(EnvCpuThreads, Environment.ProcessorCount);
        var heartbeatSeconds = ParsePositiveInt(EnvHeartbeatSeconds, 30);
        var shutdownSeconds  = ParsePositiveInt(EnvShutdownSeconds, 30);

        var healthBeacon = Environment.GetEnvironmentVariable(EnvHealthBeaconPath);
        if (string.IsNullOrWhiteSpace(healthBeacon))
        {
            healthBeacon = "/tmp/bitnet-worker-alive";
        }

        var logLevel = Environment.GetEnvironmentVariable(EnvLogLevel);
        if (string.IsNullOrWhiteSpace(logLevel))
        {
            logLevel = "info";
        }

        return new WorkerConfig(
            coordinatorUrl,
            clientId,
            clientSecret,
            workerName,
            cpuThreads,
            TimeSpan.FromSeconds(heartbeatSeconds),
            TimeSpan.FromSeconds(shutdownSeconds),
            healthBeacon,
            logLevel);
    }

    private static string RequireEnv(string name)
    {
        var value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException($"Required environment variable {name} is not set.");
        }

        return value;
    }

    private static int ParsePositiveInt(string name, int fallback)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return fallback;
        }

        if (!int.TryParse(raw, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var parsed) || parsed <= 0)
        {
            throw new InvalidOperationException($"Environment variable {name} must be a positive integer. Got '{raw}'.");
        }

        return parsed;
    }
}
