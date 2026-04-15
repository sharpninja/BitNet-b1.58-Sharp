using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Worker;

// ────────────────────────────────────────────────────────────────────────
//  BitNetSharp.Distributed.Worker entry point
// ────────────────────────────────────────────────────────────────────────
//  Worker lifecycle (Phase D-1):
//    1. Read WorkerConfig from environment variables.
//    2. Run the startup BenchmarkDotNet calibration to measure
//       throughput.
//    3. Build a CoordinatorClient, request a JWT access token from
//       the coordinator's Duende IdentityServer via OAuth 2.0
//       client credentials, and POST /register with the
//       CapabilityReport.
//    4. Log the coordinator's recommended task size.
//    5. Enter the heartbeat loop: every tick, touch the local
//       health beacon and POST a heartbeat to the coordinator so
//       the server-side sweeper keeps us Active.
//    6. Handle SIGTERM / SIGINT gracefully.
//
//  Task execution (/work poll, gradient compute, /gradient POST)
//  lands in a subsequent step — the D-1 skeleton proves the auth +
//  register + heartbeat round-trip works end-to-end first.
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

    using var client = new CoordinatorClient(config);

    Console.WriteLine($"[register] Requesting JWT from {config.CoordinatorUrl}connect/token…");
    var registration = await TryRegisterAsync(client, config, report, cts.Token).ConfigureAwait(false);

    if (registration is not null)
    {
        Console.WriteLine($"[register] Accepted as worker {registration.WorkerId}. Initial weight version {registration.InitialWeightVersion}.");
        Console.WriteLine($"[register] Coordinator-assigned task size: {registration.RecommendedTokensPerTask:N0} tokens.");
        Console.WriteLine($"[register] Heartbeat interval: {registration.HeartbeatIntervalSeconds}s.");
    }
    else
    {
        Console.WriteLine("[register] Coordinator unreachable — entering local-only mode. Health beacon still running.");
    }

    Console.WriteLine($"[heartbeat] Local beacon path: {config.HealthBeaconPath}");
    Console.WriteLine($"[heartbeat] Interval: {config.HeartbeatInterval.TotalSeconds:F0}s");

    var heartbeatTask = RunHeartbeatLoopAsync(client, config, beacon, registration is not null, cts.Token);
    var workTask = registration is not null
        ? RunWorkLoopAsync(client, config, report, cts.Token)
        : Task.CompletedTask;

    await Task.WhenAll(heartbeatTask, workTask).ConfigureAwait(false);

    Console.WriteLine("[shutdown] Heartbeat + work loops exited cleanly. Goodbye.");
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

static async Task<WorkerRegistrationResponse?> TryRegisterAsync(
    CoordinatorClient client,
    WorkerConfig config,
    CapabilityReport report,
    CancellationToken cancellationToken)
{
    var payload = new WorkerRegistrationRequest(
        WorkerName: config.WorkerName,
        EnrollmentKey: string.Empty, // OAuth flow does not use enrollment key; field kept on DTO for forward compat.
        ProcessArchitecture: RuntimeInformation.ProcessArchitecture.ToString(),
        OsDescription: RuntimeInformation.OSDescription,
        Capability: new WorkerCapabilityDto(
            TokensPerSecond: report.TokensPerSecond,
            CpuThreads: report.CpuThreads,
            CalibrationDurationMs: (long)report.CalibrationDuration.TotalMilliseconds,
            BenchmarkId: report.BenchmarkId,
            MeasuredAt: report.MeasuredAt));

    try
    {
        return await client.RegisterAsync(payload, cancellationToken).ConfigureAwait(false);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[register] Failed to register with coordinator: {ex.Message}");
        return null;
    }
}

static async Task RunWorkLoopAsync(
    CoordinatorClient client,
    WorkerConfig config,
    CapabilityReport calibration,
    CancellationToken cancellationToken)
{
    // Idle backoff when the queue is empty so a worker does not hammer
    // the coordinator with empty /work polls. 5 seconds is short
    // enough that newly enqueued tasks are picked up quickly, long
    // enough that an idle pool of a hundred workers only generates
    // 20 polls per second against the coordinator.
    var idleBackoff = TimeSpan.FromSeconds(5);

    try
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            WorkTaskAssignment? task;
            try
            {
                task = await client.TryClaimWorkAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[work] Transient error fetching task: {ex.Message}");
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            if (task is null)
            {
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            Console.WriteLine($"[work] Claimed task {task.TaskId} ({task.TokensPerTask:N0} tokens, weight v{task.WeightVersion}).");

            // D-1 stub: simulate training time proportional to the
            // calibrated throughput so the full register→work→gradient
            // round-trip can be exercised without real training. The
            // D-4 commit swaps this for actual ternary-matmul training.
            var simulatedSeconds = calibration.TokensPerSecond > 0d
                ? task.TokensPerTask / (calibration.TokensPerSecond * 0.25d)
                : 10d;
            var simulatedDuration = TimeSpan.FromSeconds(Math.Min(simulatedSeconds, 30d));
            Console.WriteLine($"[work] Simulating compute for {simulatedDuration.TotalSeconds:F1}s (stub)…");
            var wallClockStart = DateTimeOffset.UtcNow;
            try
            {
                await Task.Delay(simulatedDuration, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            var wallClockMs = (long)(DateTimeOffset.UtcNow - wallClockStart).TotalMilliseconds;
            var submission = new GradientSubmission(
                TaskId: task.TaskId,
                WorkerId: config.ClientId,
                BaseWeightVersion: task.WeightVersion,
                TokensSeen: task.TokensPerTask,
                LossAfter: 0d,                    // stub: no real loss to report
                GradientFormat: "stub-noop",
                GradientPayload: Array.Empty<byte>(),
                WallClockMs: wallClockMs);

            try
            {
                var accepted = await client.SubmitGradientAsync(submission, cancellationToken).ConfigureAwait(false);
                if (accepted)
                {
                    Console.WriteLine($"[work] Gradient submission for {task.TaskId} accepted.");
                }
                else
                {
                    Console.WriteLine($"[work] Gradient submission for {task.TaskId} rejected (ownership or stale).");
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[work] Transient error submitting gradient: {ex.Message}");
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
            }
        }
    }
    catch (OperationCanceledException)
    {
        // Expected on graceful shutdown.
    }
}

static async Task SafeDelay(TimeSpan delay, CancellationToken cancellationToken)
{
    try
    {
        await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
    }
    catch (OperationCanceledException)
    {
        // Cancellation is a normal signal to return from the caller's loop.
    }
}

static async Task RunHeartbeatLoopAsync(
    CoordinatorClient client,
    WorkerConfig config,
    HealthBeacon beacon,
    bool registered,
    CancellationToken cancellationToken)
{
    using var timer = new PeriodicTimer(config.HeartbeatInterval);
    try
    {
        while (await timer.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
        {
            beacon.Touch();

            if (!registered)
            {
                continue;
            }

            try
            {
                var heartbeat = new HeartbeatRequest(
                    WorkerId: config.ClientId,
                    Status: "idle",
                    CurrentTaskId: null,
                    TokensSeenSinceLastHeartbeat: 0);

                var response = await client.SendHeartbeatAsync(heartbeat, cancellationToken).ConfigureAwait(false);
                if (response is null)
                {
                    Console.Error.WriteLine("[heartbeat] Coordinator returned 410 Gone. Worker should restart to re-register.");
                }
                else if (response.ShouldDrain)
                {
                    Console.WriteLine("[heartbeat] Drain requested by coordinator. Exiting.");
                    return;
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[heartbeat] Transient error: {ex.Message}");
            }
        }
    }
    catch (OperationCanceledException)
    {
        // Expected on graceful shutdown.
    }
}

static void PrintBanner(WorkerConfig config)
{
    var culture = CultureInfo.InvariantCulture;
    Console.WriteLine("───────────────────────────────────────────────────────────────");
    Console.WriteLine(" BitNetSharp.Distributed.Worker");
    Console.WriteLine("───────────────────────────────────────────────────────────────");
    Console.WriteLine(string.Create(culture, $" worker name         : {config.WorkerName}"));
    Console.WriteLine(string.Create(culture, $" coordinator         : {config.CoordinatorUrl}"));
    Console.WriteLine(string.Create(culture, $" client id           : {config.ClientId}"));
    Console.WriteLine(string.Create(culture, $" cpu threads         : {config.CpuThreads}"));
    Console.WriteLine(string.Create(culture, $" process architecture: {RuntimeInformation.ProcessArchitecture}"));
    Console.WriteLine(string.Create(culture, $" os description      : {RuntimeInformation.OSDescription}"));
    Console.WriteLine(string.Create(culture, $" heartbeat interval  : {config.HeartbeatInterval.TotalSeconds:F0}s"));
    Console.WriteLine(string.Create(culture, $" shutdown grace      : {config.ShutdownGrace.TotalSeconds:F0}s"));
    Console.WriteLine(string.Create(culture, $" log level           : {config.LogLevel}"));
    Console.WriteLine("───────────────────────────────────────────────────────────────");
}

static void WireShutdownSignals(CancellationTokenSource cts, TimeSpan grace)
{
    Console.CancelKeyPress += (_, e) =>
    {
        e.Cancel = true;
        Console.WriteLine("[signal] Ctrl+C received; beginning graceful shutdown.");
        cts.Cancel();
    };

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
