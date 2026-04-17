using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Worker;
using Serilog;
using Serilog.Events;
using Serilog.Sinks.PeriodicBatching;

// ────────────────────────────────────────────────────────────────────────
//  BitNetSharp.Distributed.Worker entry point
// ────────────────────────────────────────────────────────────────────────
//  Worker lifecycle (Phase D-1):
//    1. Read WorkerConfig from environment variables.
//    2. Run the startup BenchmarkDotNet calibration to measure
//       throughput.
//    3. Build a CoordinatorClient (which seeds the X-Api-Key +
//       X-Worker-Id request headers from the env config) and POST
//       /register with the CapabilityReport.
//    4. Log the coordinator's recommended task size.
//    5. Enter the heartbeat loop: every tick, touch the local
//       health beacon and POST a heartbeat to the coordinator so
//       the server-side sweeper keeps us Active.
//    6. Handle SIGTERM / SIGINT gracefully.
//
//  Worker auth = single shared API key set by the operator on the
//  coordinator via `Coordinator__WorkerApiKey` env var. Every worker
//  presents the same key in `X-Api-Key`; a rotate = change the env
//  var + restart coordinator and every worker with the old key is
//  locked out instantly.
// ────────────────────────────────────────────────────────────────────────

// ── Serilog bootstrap ─────────────────────────────────────────────
// The coordinator log sink is wired up early so it can capture
// startup messages. It drops batches silently until SetClient is
// called after authentication succeeds; the console sink covers
// the gap.
var coordinatorSink = new CoordinatorLogSink();
var batchingOptions = new PeriodicBatchingSinkOptions
{
    BatchSizeLimit = 50,
    Period = TimeSpan.FromSeconds(5)
};
var batchingSink = new PeriodicBatchingSink(coordinatorSink, batchingOptions);

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
    .WriteTo.Sink(batchingSink)
    .CreateLogger();

try
{
    var config = WorkerConfig.FromEnvironment();
    PrintBanner(config);

    using var cts = new CancellationTokenSource();
    WireShutdownSignals(cts, config.ShutdownGrace);

    var beacon = new HealthBeacon(config.HealthBeaconPath);

    Log.Information("Running startup BenchmarkDotNet capability pass (usually 15-45s)…");
    var report = StartupCalibrator.Run(config);
    Log.Information("{Display}", report.ToDisplayString());

    using var client = new CoordinatorClient(config);

    Log.Information("Registering with coordinator at {Url} as worker id {WorkerId}…",
        config.CoordinatorUrl, config.WorkerId);
    var registration = await TryRegisterAsync(client, config, report, cts.Token).ConfigureAwait(false);

    if (registration is not null)
    {
        // Hand the client to the Serilog sink so it can start
        // shipping log batches to the coordinator. Auth headers are
        // attached by the client itself so the sink does not need
        // to touch tokens or secrets.
        coordinatorSink.SetClient(client);

        Log.Information("Accepted as worker {WorkerId}. Initial weight version {Version}. Task size {Tokens:N0} tokens. Heartbeat {Heartbeat}s.",
            registration.WorkerId,
            registration.InitialWeightVersion,
            registration.RecommendedTokensPerTask,
            registration.HeartbeatIntervalSeconds);
    }
    else
    {
        Log.Warning("Coordinator unreachable — entering local-only mode. Health beacon still running.");
    }

    Log.Information("Local beacon path: {BeaconPath}. Interval: {Interval}s.",
        config.HealthBeaconPath, config.HeartbeatInterval.TotalSeconds);

    var heartbeatTask = RunHeartbeatLoopAsync(client, config, beacon, registration is not null, cts.Token);
    var workTask = registration is not null
        ? RunWorkLoopAsync(client, config, report, cts.Token)
        : Task.CompletedTask;

    await Task.WhenAll(heartbeatTask, workTask).ConfigureAwait(false);

    Log.Information("Heartbeat + work loops exited cleanly. Goodbye.");
    return 0;
}
catch (InvalidOperationException configError)
{
    Log.Fatal(configError, "Configuration error");
    return 2;
}
catch (Exception unexpected)
{
    Log.Fatal(unexpected, "Unhandled exception");
    return 1;
}
finally
{
    await Log.CloseAndFlushAsync().ConfigureAwait(false);
}

static async Task<WorkerRegistrationResponse?> TryRegisterAsync(
    CoordinatorClient client,
    WorkerConfig config,
    CapabilityReport report,
    CancellationToken cancellationToken)
{
    var payload = new WorkerRegistrationRequest(
        WorkerName: config.WorkerName,
        EnrollmentKey: string.Empty, // Shared-key flow does not use enrollment key; field kept on DTO for forward compat.
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
        Log.Error(ex, "Failed to register with coordinator");
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

    // Error-feedback residual state maintained across tasks. The
    // int8 quantization residual from each step is added into the
    // next step's gradient before encoding so quantization bias
    // corrects over time. Reset when gradient dimension changes.
    float[]? errorFeedbackResidual = null;

    // Cached copy of the global weight vector keyed by version so a
    // worker that runs 20 consecutive tasks against the same weight
    // version downloads the blob exactly once. The dimension of the
    // cached array determines the gradient dimension the worker
    // produces — crucial to match the coordinator or /gradient 400s.
    float[]? cachedWeights = null;
    long cachedWeightsVersion = -1;

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
                Log.Warning(ex, "Transient error fetching task");
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            if (task is null)
            {
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            Log.Information("Claimed task {TaskId} ({Tokens:N0} tokens, weight v{Version})", task.TaskId, task.TokensPerTask, task.WeightVersion);

            // D-4b: compute a real gradient against the current
            // weight vector, encode it with int8 error feedback, and
            // submit the encoded payload. The "gradient" is a simple
            // convergence-test stub: g = (target - current) scaled so
            // the global vector approaches the target over many
            // rounds. Real BitNet backprop replaces this in Phase A.
            var wallClockStart = DateTimeOffset.UtcNow;
            Log.Debug("Computing gradient for task {TaskId} (D-4b stub)", task.TaskId);

            // Download the current weight snapshot from the coordinator
            // when the task's WeightVersion differs from what we have
            // cached. Decoding the blob gives us the authoritative
            // dimension so the encoded gradient shape matches what the
            // coordinator expects in its /gradient handler.
            if (cachedWeights is null || cachedWeightsVersion != task.WeightVersion)
            {
                try
                {
                    var blob = await client
                        .DownloadWeightsAsync(task.WeightUrl, cancellationToken)
                        .ConfigureAwait(false);
                    if (!WeightBlobCodec.TryDecode(blob, out var blobVersion, out var weights, out var decodeErr))
                    {
                        Log.Warning(
                            "Weight blob at {Url} failed to decode: {Err}. Skipping task.",
                            task.WeightUrl,
                            decodeErr);
                        await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                        continue;
                    }

                    cachedWeights = weights;
                    cachedWeightsVersion = blobVersion;
                    Log.Information(
                        "Downloaded weight version {Version} ({Dim:N0} elements) from {Url}",
                        blobVersion,
                        weights.Length,
                        task.WeightUrl);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Log.Warning(ex, "Transient error downloading weights");
                    await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                    continue;
                }
            }

            var dim = cachedWeights.Length;
            var currentWeights = cachedWeights;

            // Synthetic gradient: push weights toward a constant
            // target so the coordinator's weight vector converges
            // visibly in the dashboard's loss trace.
            var target = new float[dim];
            var rng = new Random(task.TaskId.GetHashCode());
            for (var wi = 0; wi < dim; wi++)
            {
                target[wi] = (float)(rng.NextDouble() * 2d - 1d);
            }

            var gradient = new float[dim];
            for (var wi = 0; wi < dim; wi++)
            {
                gradient[wi] = currentWeights[wi] - target[wi];
            }

            // Int8 error-feedback encoding — maintain a per-worker
            // residual across tasks so quantization bias corrects
            // over time. The residual is reset when the gradient
            // dimension changes (e.g. after a model resize).
            if (errorFeedbackResidual is null || errorFeedbackResidual.Length != dim)
            {
                errorFeedbackResidual = new float[dim];
            }

            // Add prior residual to current gradient before encoding.
            for (var wi = 0; wi < dim; wi++)
            {
                gradient[wi] += errorFeedbackResidual[wi];
            }

            var payload = Int8GradientCodec.Encode(gradient, errorFeedbackResidual);
            var wallClockMs = (long)(DateTimeOffset.UtcNow - wallClockStart).TotalMilliseconds;

            // Compute a crude "loss" as mean squared distance to target.
            var loss = 0d;
            for (var wi = 0; wi < dim; wi++)
            {
                var diff = currentWeights[wi] - target[wi];
                loss += diff * diff;
            }
            loss /= dim;

            var submission = new GradientSubmission(
                TaskId: task.TaskId,
                WorkerId: config.WorkerId,
                BaseWeightVersion: task.WeightVersion,
                TokensSeen: task.TokensPerTask,
                LossAfter: loss,
                GradientFormat: Int8GradientCodec.FormatId,
                GradientPayload: payload,
                WallClockMs: wallClockMs);

            try
            {
                var accepted = await client.SubmitGradientAsync(submission, cancellationToken).ConfigureAwait(false);
                if (accepted)
                {
                    Log.Information("Gradient for {TaskId} accepted", task.TaskId);
                }
                else
                {
                    Log.Warning("Gradient for {TaskId} rejected (ownership or stale)", task.TaskId);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "Transient error submitting gradient");
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
                    WorkerId: config.WorkerId,
                    Status: "idle",
                    CurrentTaskId: null,
                    TokensSeenSinceLastHeartbeat: 0);

                var response = await client.SendHeartbeatAsync(heartbeat, cancellationToken).ConfigureAwait(false);
                if (response is null)
                {
                    Log.Error("Coordinator returned 410 Gone. Worker should restart to re-register.");
                }
                else if (response.ShouldDrain)
                {
                    Log.Information("Drain requested by coordinator. Exiting.");
                    return;
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                Log.Warning(ex, "Transient heartbeat error");
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
    Console.WriteLine(string.Create(culture, $" worker id           : {config.WorkerId}"));
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
        Log.Information("Ctrl+C received; beginning graceful shutdown.");
        cts.Cancel();
    };

    AppDomain.CurrentDomain.ProcessExit += (_, _) =>
    {
        Log.Information("SIGTERM received; beginning graceful shutdown.");
        cts.Cancel();
        var deadline = DateTime.UtcNow + grace;
        while (DateTime.UtcNow < deadline)
        {
            Thread.Sleep(50);
        }
    };
}
