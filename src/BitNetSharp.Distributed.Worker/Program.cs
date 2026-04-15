using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Worker;

// ────────────────────────────────────────────────────────────────────────
//  BitNetSharp.Distributed.Worker entry point
// ────────────────────────────────────────────────────────────────────────
//  Responsibilities in the v1 (Phase D-1) skeleton:
//    1. Read WorkerConfig from environment variables (fail fast on missing).
//    2. Run the startup BenchmarkDotNet calibration pass to measure how
//       fast THIS hardware can chew through the representative ternary
//       matmul hot loop.
//    3. Emit a CapabilityReport so the coordinator can size task batches
//       to ~10 minutes of compute on this worker.
//    4. Register with the coordinator (stubbed: Phase D-1 will replace the
//       log line with an actual HTTP POST to /register).
//    5. Start a heartbeat + health-beacon loop so Docker healthchecks,
//       orchestrators, and the coordinator all know the worker is alive.
//    6. Shut down gracefully on SIGTERM / SIGINT so orchestrators can
//       reclaim instances without poisoning the queue.
// ────────────────────────────────────────────────────────────────────────

try
{
    var config = WorkerConfig.FromEnvironment();
    PrintBanner(config);

    using var cts = new CancellationTokenSource();
    WireShutdownSignals(cts, config.ShutdownGrace);

    var beacon = new HealthBeacon(config.HealthBeaconPath);

    Console.WriteLine("[calibrate] Running startup BenchmarkDotNet capability pass (this usually takes ~15–45 seconds)…");
    var report = StartupCalibrator.Run(config);
    Console.WriteLine($"[calibrate] {report.ToDisplayString()}");

    // Coordinator integration is stubbed for Phase D-1 scaffolding. The
    // follow-up work item wires this to an HTTP client that POSTs the
    // CapabilityReport to {CoordinatorUrl}/register and then long-polls
    // {CoordinatorUrl}/work for task assignments.
    Console.WriteLine($"[register] Would POST CapabilityReport to {config.CoordinatorUrl}register (Phase D-1).");
    Console.WriteLine($"[register] Recommended task size: {report.RecommendedTokensPerTask():N0} tokens per task.");

    Console.WriteLine($"[heartbeat] Health beacon path: {config.HealthBeaconPath}");
    Console.WriteLine($"[heartbeat] Interval: {config.HeartbeatInterval.TotalSeconds:F0}s");

    await beacon.RunAsync(config.HeartbeatInterval, cts.Token).ConfigureAwait(false);

    Console.WriteLine("[shutdown] Heartbeat loop exited cleanly. Goodbye.");
    return 0;
}
catch (InvalidOperationException configError)
{
    Console.Error.WriteLine($"[fatal] Configuration error: {configError.Message}");
    return 2;
}
catch (Exception unexpected)
{
    Console.Error.WriteLine($"[fatal] Unhandled exception: {unexpected}");
    return 1;
}

static void PrintBanner(WorkerConfig config)
{
    var culture = CultureInfo.InvariantCulture;
    Console.WriteLine("───────────────────────────────────────────────────────────────");
    Console.WriteLine(" BitNetSharp.Distributed.Worker");
    Console.WriteLine("───────────────────────────────────────────────────────────────");
    Console.WriteLine(string.Create(culture, $" worker name        : {config.WorkerName}"));
    Console.WriteLine(string.Create(culture, $" coordinator        : {config.CoordinatorUrl}"));
    Console.WriteLine(string.Create(culture, $" cpu threads        : {config.CpuThreads}"));
    Console.WriteLine(string.Create(culture, $" process architecture: {RuntimeInformation.ProcessArchitecture}"));
    Console.WriteLine(string.Create(culture, $" os description     : {RuntimeInformation.OSDescription}"));
    Console.WriteLine(string.Create(culture, $" heartbeat interval : {config.HeartbeatInterval.TotalSeconds:F0}s"));
    Console.WriteLine(string.Create(culture, $" shutdown grace     : {config.ShutdownGrace.TotalSeconds:F0}s"));
    Console.WriteLine(string.Create(culture, $" log level          : {config.LogLevel}"));
    Console.WriteLine("───────────────────────────────────────────────────────────────");
}

static void WireShutdownSignals(CancellationTokenSource cts, TimeSpan grace)
{
    // Ctrl+C during local development.
    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        Console.WriteLine("[signal] Ctrl+C received; beginning graceful shutdown.");
        cts.Cancel();
    };

    // SIGTERM from Docker / systemd / orchestrators. On .NET 10 this is
    // delivered via AppDomain.ProcessExit which runs synchronously, so we
    // cancel the token and then block for the grace window so in-flight
    // heartbeat writes can drain before the runtime tears the process down.
    AppDomain.CurrentDomain.ProcessExit += (_, _) =>
    {
        Console.WriteLine("[signal] SIGTERM received; beginning graceful shutdown.");
        cts.Cancel();
        var deadline = DateTime.UtcNow + grace;
        while (DateTime.UtcNow < deadline)
        {
            Thread.Sleep(50);
        }
    };
}
