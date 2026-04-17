using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
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

    // Dedicated HTTP client for corpus shard fetches. Same auth
    // headers as the coordinator client; separate because the corpus
    // response bodies are multi-MiB and we do not want them to block
    // or time-share the worker's registration / heartbeat / gradient
    // submissions on the same connection.
    using var corpusHttp = new System.Net.Http.HttpClient
    {
        BaseAddress = config.CoordinatorUrl,
        Timeout = TimeSpan.FromMinutes(2)
    };
    corpusHttp.DefaultRequestHeaders.Add(CoordinatorClient.ApiKeyHeader, config.ApiKey);
    corpusHttp.DefaultRequestHeaders.Add(CoordinatorClient.WorkerIdHeader, config.WorkerId);
    using var corpusClient = new CorpusClient(corpusHttp, ownsHttpClient: false, maxCachedShards: 2);

    // Hand the client to the Serilog sink immediately so log batches
    // can flow as soon as auth is valid. The sink only ships when it
    // holds a non-null client; the initial registration supervisor
    // may take multiple retries on a cold coordinator.
    coordinatorSink.SetClient(client);

    var gate = new RegistrationGate();

    Log.Information("Local beacon path: {BeaconPath}. Interval: {Interval}s.",
        config.HealthBeaconPath, config.HeartbeatInterval.TotalSeconds);

    // Build the BitNetConfig that matches the coordinator's global
    // model from the operator-selected preset. ComputeLength of this
    // config is the flat-parameter vector length the worker is
    // prepared to train against; /weights blobs whose length does
    // not match will be skipped with a warning until the coordinator
    // side is upgraded to serve full flat params.
    var modelConfig = BuildModelConfig(config.ModelPreset);
    Log.Information(
        "Model preset: {Preset} (vocab={Vocab}, dim={Dim}, hidden={Hidden}, layers={Layers}, seq={Seq}) — expected flat-param length {Len:N0}",
        config.ModelPreset,
        modelConfig.VocabSize,
        modelConfig.Dimension,
        modelConfig.HiddenDimension,
        modelConfig.LayerCount,
        modelConfig.MaxSequenceLength,
        FlatParameterPack.ComputeLength(modelConfig));

    var supervisorTask = RunRegistrationSupervisorAsync(client, config, report, gate, cts.Token);
    var heartbeatTask  = RunHeartbeatLoopAsync(client, config, beacon, gate, cts.Token);
    var workTask       = RunWorkLoopAsync(client, corpusClient, config, modelConfig, gate, cts.Token);

    await Task.WhenAll(supervisorTask, heartbeatTask, workTask).ConfigureAwait(false);

    Log.Information("Supervisor + heartbeat + work loops exited cleanly. Goodbye.");
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

static async Task RunRegistrationSupervisorAsync(
    CoordinatorClient client,
    WorkerConfig config,
    CapabilityReport report,
    RegistrationGate gate,
    CancellationToken cancellationToken)
{
    // Exponential backoff so a cold-booted worker hammering a dead
    // coordinator escalates quickly from "try again in 2 s" to
    // "try again in 30 s" without ever giving up. When registration
    // finally succeeds the backoff resets to the floor so the NEXT
    // outage starts from a fresh 2 s retry again.
    var minDelay = TimeSpan.FromSeconds(2);
    var maxDelay = TimeSpan.FromSeconds(30);
    var delay    = minDelay;

    try
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            if (!gate.IsRegistered)
            {
                Log.Information(
                    "Registering with coordinator at {Url} as worker {WorkerId}…",
                    config.CoordinatorUrl,
                    config.WorkerId);

                var registration = await TryRegisterOnceAsync(client, config, report, cancellationToken)
                    .ConfigureAwait(false);

                if (registration is not null)
                {
                    gate.MarkRegistered();
                    delay = minDelay;
                    Log.Information(
                        "Accepted as worker {WorkerId}. Weight v{Version}. Task size {Tokens:N0}. Heartbeat {Heartbeat}s.",
                        registration.WorkerId,
                        registration.InitialWeightVersion,
                        registration.RecommendedTokensPerTask,
                        registration.HeartbeatIntervalSeconds);
                }
                else
                {
                    Log.Warning(
                        "Registration attempt failed. Retrying in {Delay:F0}s (coordinator may be restarting).",
                        delay.TotalSeconds);
                    await SafeDelay(delay, cancellationToken).ConfigureAwait(false);
                    delay = TimeSpan.FromSeconds(Math.Min(maxDelay.TotalSeconds, delay.TotalSeconds * 1.5d));
                    continue;
                }
            }

            // Registered — park until a loss signal arrives.
            await gate.WaitForLossAsync(cancellationToken).ConfigureAwait(false);
            Log.Warning("Registration lost; re-registering.");
        }
    }
    catch (OperationCanceledException)
    {
        // Expected on graceful shutdown.
    }
}

static async Task<WorkerRegistrationResponse?> TryRegisterOnceAsync(
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
    catch (OperationCanceledException)
    {
        throw;
    }
    catch (Exception ex)
    {
        // Do not log a full stack trace every retry — the supervisor
        // will try again shortly and the spam obscures the real
        // recovery moment. Single-line INFO is enough.
        Log.Warning("Register failed: {Message}", ex.Message);
        return null;
    }
}

static async Task RunWorkLoopAsync(
    CoordinatorClient client,
    CorpusClient corpusClient,
    WorkerConfig config,
    BitNetConfig modelConfig,
    RegistrationGate gate,
    CancellationToken cancellationToken)
{
    var expectedFlatLength = FlatParameterPack.ComputeLength(modelConfig);
    // When the coordinator's corpus route cannot satisfy a shard
    // fetch, fall back to deterministic synthesized tokens ONLY if
    // the operator has explicitly opted in. Default is strict:
    // skip the task rather than train on random data. This matches
    // the Phase A policy that a worker must never corrupt weights
    // with noise it could not verify came from the real corpus.
    var allowSyntheticShards = string.Equals(
        Environment.GetEnvironmentVariable("BITNET_ALLOW_SYNTHETIC_SHARDS"),
        "true",
        StringComparison.OrdinalIgnoreCase);
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

    // Counter for consecutive /work failures. After too many in a
    // row we flip the registration gate to Lost so the supervisor
    // re-registers; this recovers from the case where the
    // coordinator comes back with a wiped worker registry.
    var consecutiveWorkFailures = 0;
    const int FailuresBeforeMarkingLost = 5;

    try
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            // Park the work loop while the gate says we are not
            // registered; the supervisor drives re-registration.
            if (!gate.IsRegistered)
            {
                await SafeDelay(TimeSpan.FromSeconds(1), cancellationToken).ConfigureAwait(false);
                continue;
            }

            WorkTaskAssignment? task;
            try
            {
                task = await client.TryClaimWorkAsync(cancellationToken).ConfigureAwait(false);
                consecutiveWorkFailures = 0;
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                consecutiveWorkFailures++;
                Log.Warning(
                    "Transient error fetching task ({Count}/{Limit}): {Message}",
                    consecutiveWorkFailures,
                    FailuresBeforeMarkingLost,
                    ex.Message);
                if (consecutiveWorkFailures >= FailuresBeforeMarkingLost)
                {
                    Log.Warning("Coordinator unreachable for {Count} polls — flipping to unregistered so supervisor re-registers.",
                        consecutiveWorkFailures);
                    gate.MarkLost();
                    consecutiveWorkFailures = 0;
                }
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            if (task is null)
            {
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            Log.Information("Claimed task {TaskId} ({Tokens:N0} tokens, weight v{Version})", task.TaskId, task.TokensPerTask, task.WeightVersion);

            // Phase A Track 5: compute a real BitNet training
            // gradient against the current weight vector, encode it
            // with int8 error feedback, and submit the encoded
            // payload. The "gradient" is the delta between locally-
            // trained master parameters and the assigned snapshot
            // (new_flat - old_flat). The coordinator aggregates these
            // deltas across workers and advances the weight version.
            var wallClockStart = DateTimeOffset.UtcNow;
            Log.Debug("Computing gradient for task {TaskId}", task.TaskId);

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

            // Guard against a coordinator that has not yet been
            // upgraded to serve full flat params (e.g. the legacy
            // 4096-element placeholder). We cannot train against a
            // vector of the wrong shape, so skip the task with a
            // warning rather than corrupting the global weights.
            if (dim != expectedFlatLength)
            {
                Log.Warning(
                    "Weight vector length {Got} does not match expected {Expected} for preset {Preset}. "
                    + "Coordinator has not been upgraded to the Phase A flat-parameter protocol yet — skipping task {TaskId}.",
                    dim,
                    expectedFlatLength,
                    config.ModelPreset,
                    task.TaskId);
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            // Fetch real shard bytes for this task from the
            // coordinator's /corpus/{shardId} endpoint via HTTP Range.
            // Fall back to deterministic synthesis only if the
            // operator has opted in via BITNET_ALLOW_SYNTHETIC_SHARDS=
            // true — corrupting the global weights with random data
            // when the real corpus fails to stream is worse than
            // idling the worker.
            IReadOnlyList<int[]> shardSequences;
            try
            {
                shardSequences = await corpusClient
                    .FetchSequencesAsync(task, modelConfig.MaxSequenceLength, cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                if (allowSyntheticShards)
                {
                    Log.Warning(
                        ex,
                        "Real shard fetch failed for task {TaskId} (shard {ShardId}); "
                        + "falling back to synthetic tokens because BITNET_ALLOW_SYNTHETIC_SHARDS=true",
                        task.TaskId, task.ShardId);
                    shardSequences = SynthesizeShardTokens(task, modelConfig);
                }
                else
                {
                    Log.Warning(
                        "Real shard fetch failed for task {TaskId} (shard {ShardId}): {Message}. "
                        + "Skipping task (set BITNET_ALLOW_SYNTHETIC_SHARDS=true to fall back to synth).",
                        task.TaskId, task.ShardId, ex.Message);
                    await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                    continue;
                }
            }

            if (shardSequences.Count == 0)
            {
                if (allowSyntheticShards)
                {
                    Log.Warning(
                        "Shard {ShardId}@{Offset}+{Length} yielded no full sequences at seqLen={Seq}; "
                        + "falling back to synthetic tokens",
                        task.ShardId, task.ShardOffset, task.ShardLength, modelConfig.MaxSequenceLength);
                    shardSequences = SynthesizeShardTokens(task, modelConfig);
                }
                else
                {
                    Log.Warning(
                        "Shard {ShardId}@{Offset}+{Length} yielded no full sequences at seqLen={Seq}. "
                        + "Skipping task.",
                        task.ShardId, task.ShardOffset, task.ShardLength, modelConfig.MaxSequenceLength);
                    await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                    continue;
                }
            }

            var totalTokens = 0;
            foreach (var seq in shardSequences)
            {
                totalTokens += seq.Length;
            }

            float[] gradient;
            try
            {
                gradient = RealTrainingGradient.ComputeGradient(
                    currentWeights,
                    shardSequences,
                    modelConfig,
                    localSteps: Math.Max(1, task.KLocalSteps));
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Real-training gradient computation failed for task {TaskId}", task.TaskId);
                await SafeDelay(idleBackoff, cancellationToken).ConfigureAwait(false);
                continue;
            }

            Log.Information(
                "Computed gradient for task {TaskId} (Phase A training, {Steps} local steps, {Tokens} tok)",
                task.TaskId,
                Math.Max(1, task.KLocalSteps),
                totalTokens);

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

            // Crude "loss" proxy: mean-squared norm of the weight
            // delta. Real training loss is reported inside the
            // trainer's Console logs; the coordinator only needs a
            // single scalar for its dashboard trendline.
            var loss = 0d;
            for (var wi = 0; wi < dim; wi++)
            {
                loss += gradient[wi] * (double)gradient[wi];
            }
            loss /= dim;

            var measuredTps = wallClockMs > 0
                ? task.TokensPerTask / (wallClockMs / 1000.0)
                : (double?)null;

            var submission = new GradientSubmission(
                TaskId: task.TaskId,
                WorkerId: config.WorkerId,
                BaseWeightVersion: task.WeightVersion,
                TokensSeen: task.TokensPerTask,
                LossAfter: loss,
                GradientFormat: Int8GradientCodec.FormatId,
                GradientPayload: payload,
                WallClockMs: wallClockMs,
                MeasuredTokensPerSecond: measuredTps);

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
    RegistrationGate gate,
    CancellationToken cancellationToken)
{
    using var timer = new PeriodicTimer(config.HeartbeatInterval);
    try
    {
        while (await timer.WaitForNextTickAsync(cancellationToken).ConfigureAwait(false))
        {
            beacon.Touch();

            if (!gate.IsRegistered)
            {
                // Supervisor owns registration; heartbeat is a no-op
                // while we wait for it to land.
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
                    // 410 Gone — the coordinator wiped the row, e.g.
                    // because it restarted with an empty registry.
                    // Flip the gate so the supervisor re-registers.
                    Log.Warning("Coordinator returned 410 Gone. Re-registering.");
                    gate.MarkLost();
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
                // Network-level heartbeat errors are common during a
                // coordinator restart; the work loop's failure
                // counter will flip the gate if they persist.
                Log.Warning("Transient heartbeat error: {Message}", ex.Message);
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
    Console.WriteLine(string.Create(culture, $" model preset        : {config.ModelPreset}"));
    Console.WriteLine("───────────────────────────────────────────────────────────────");
}

static BitNetConfig BuildModelConfig(string presetName)
{
    // TruckMateModelPresets is the single source of truth for
    // named preset sizes (small / medium / large). Build a
    // BitNetConfig that matches so FlatParameterPack.ComputeLength
    // agrees with the coordinator's notion of the global model.
    var preset = TruckMateModelPresets.GetPreset(presetName);
    return new BitNetConfig(
        vocabSize: preset.VocabSize,
        dimension: preset.Dimension,
        hiddenDimension: preset.HiddenDimension,
        layerCount: preset.LayerCount,
        headCount: preset.HeadCount,
        maxSequenceLength: preset.MaxSequenceLength);
}

static IReadOnlyList<int[]> SynthesizeShardTokens(WorkTaskAssignment task, BitNetConfig modelConfig)
{
    // Until the coordinator ships real shard bytes with each task,
    // synthesize a deterministic token stream from the task's
    // ShardId / ShardOffset so each worker trains on a reproducible
    // but non-degenerate corpus. This matches the wire-format
    // expectation that the coordinator's corpus generator ships a
    // flat little-endian int32 token stream.
    var seed = HashCode.Combine(task.ShardId, task.ShardOffset, task.ShardLength);
    var rng = new Random(seed);

    // Cap token count at a fraction of the task budget so a single
    // /work round does not run for minutes. One sequence per
    // maxSequenceLength chunk, at most 4 sequences per task.
    var maxSeq = Math.Max(2, modelConfig.MaxSequenceLength);
    var targetTokens = (int)Math.Min(task.TokensPerTask, 4L * maxSeq);
    var sequenceCount = Math.Max(1, targetTokens / maxSeq);

    var sequences = new List<int[]>(sequenceCount);
    for (var s = 0; s < sequenceCount; s++)
    {
        var seq = new int[maxSeq];
        for (var t = 0; t < seq.Length; t++)
        {
            seq[t] = rng.Next(0, modelConfig.VocabSize);
        }
        sequences.Add(seq);
    }

    return sequences;
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
