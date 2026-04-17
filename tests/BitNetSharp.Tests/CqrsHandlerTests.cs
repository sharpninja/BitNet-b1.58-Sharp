#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Cqrs.Commands;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
using BitNetSharp.Distributed.Coordinator.Services;
using GetTaskQueueSnapshotQueryHandler = BitNetSharp.Distributed.Coordinator.Cqrs.Queries.GetTaskQueueSnapshotQueryHandler;
using BitNetSharp.Distributed.Coordinator.Persistence;
using McpServer.Cqrs;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Isolated handler tests for every CQRS command / query the
/// coordinator exposes. Each test instantiates the real store layer
/// against a temp SQLite file, wires the handler directly, and
/// dispatches through a hand-rolled <see cref="CallContext"/> —
/// the Dispatcher pipeline is exercised separately in the
/// CoordinatorEndpointTests integration suite.
/// </summary>
public sealed class CqrsHandlerTests : IDisposable
{
    private readonly string _databasePath;
    private readonly string _weightsDirectory;
    private readonly FakeTimeProvider _time;
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteWorkQueueStore _queueStore;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly FileSystemWeightStore _weightStore;
    private readonly WeightApplicationService _weightApplication;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;

    public CqrsHandlerTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-cqrs-{Guid.NewGuid():N}.db");
        _weightsDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-cqrs-weights-{Guid.NewGuid():N}");
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 19, 0, 0, TimeSpan.Zero));
        var connectionString = $"Data Source={_databasePath}";
        _workerStore = new SqliteWorkerRegistryStore(connectionString, _time);
        _queueStore = new SqliteWorkQueueStore(connectionString, _time);
        _telemetry = new SqliteTelemetryStore(connectionString, _time);
        _weightStore = new FileSystemWeightStore(_weightsDirectory);

        _options = new StaticOptionsMonitor<CoordinatorOptions>(new CoordinatorOptions
        {
            TargetTaskDurationSeconds = 600,
            FullStepEfficiency = 0.25d,
            HeartbeatIntervalSeconds = 30,
            InitialWeightVersion = 1,
            ModelPreset = "", InitialWeightDimension = 8,
            BaseLearningRate = 0.1d,
            StalenessAlpha = 0.5d,
            MaxStalenessSteps = 5,
            BaseUrl = "http://localhost"
        });

        _weightApplication = new WeightApplicationService(
            _weightStore,
            _options,
            NullLogger<WeightApplicationService>.Instance);
        _weightApplication.EnsureInitialized();
    }

    public void Dispose()
    {
        _workerStore.Dispose();
        _queueStore.Dispose();
        _telemetry.Dispose();
        TryDelete(_databasePath);
        TryDelete(_databasePath + "-wal");
        TryDelete(_databasePath + "-shm");
        if (Directory.Exists(_weightsDirectory))
        {
            try { Directory.Delete(_weightsDirectory, recursive: true); } catch { }
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
    }

    private WorkTaskRecord NewPendingTask(string id) => new(
        TaskId: id,
        WeightVersion: 1,
        ShardId: "shard-A",
        ShardOffset: 0,
        ShardLength: 1024,
        TokensPerTask: 4096,
        KLocalSteps: 4,
        HyperparametersJson: "{}",
        State: WorkTaskState.Pending,
        AssignedWorkerId: null,
        AssignedAtUtc: null,
        DeadlineUtc: null,
        Attempt: 0,
        CreatedAtUtc: _time.GetUtcNow(),
        CompletedAtUtc: null);

    // ── RegisterWorkerCommand ───────────────────────────────────────

    [Fact]
    public async Task RegisterWorker_creates_worker_row_and_returns_recommended_tokens()
    {
        var handler = new RegisterWorkerCommandHandler(
            _workerStore,
            _options,
            _time,
            NullLogger<RegisterWorkerCommandHandler>.Instance);

        var command = new RegisterWorkerCommand(
            ClientId: "worker-alpha",
            Request: new WorkerRegistrationRequest(
                WorkerName: "PAYTON-LEGION2",
                EnrollmentKey: string.Empty,
                ProcessArchitecture: "X64",
                OsDescription: "TestOS",
                Capability: new WorkerCapabilityDto(
                    TokensPerSecond: 1000d,
                    CpuThreads: 4,
                    CalibrationDurationMs: 1234,
                    BenchmarkId: "Int8TernaryMatMul",
                    MeasuredAt: _time.GetUtcNow())));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsSuccess);
        Assert.Equal("worker-alpha", result.Value!.WorkerId);
        Assert.Equal(150_016L, result.Value.RecommendedTokensPerTask);
        Assert.Equal(1, result.Value.InitialWeightVersion);
        Assert.Equal(30, result.Value.HeartbeatIntervalSeconds);
        Assert.NotNull(_workerStore.FindById("worker-alpha"));
    }

    [Fact]
    public async Task RegisterWorker_fails_when_client_id_empty()
    {
        var handler = new RegisterWorkerCommandHandler(
            _workerStore,
            _options,
            _time,
            NullLogger<RegisterWorkerCommandHandler>.Instance);

        var command = new RegisterWorkerCommand(
            ClientId: string.Empty,
            Request: new WorkerRegistrationRequest(
                WorkerName: "x",
                EnrollmentKey: string.Empty,
                ProcessArchitecture: "X64",
                OsDescription: "TestOS",
                Capability: new WorkerCapabilityDto(1000d, 4, 1234, "x", _time.GetUtcNow())));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
    }

    // ── ClaimNextTaskCommand ────────────────────────────────────────

    [Fact]
    public async Task ClaimNextTask_returns_null_when_queue_is_empty()
    {
        var handler = new ClaimNextTaskCommandHandler(_queueStore, _options, _weightApplication, _telemetry, _time);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new ClaimNextTaskCommand("worker-alpha"), context);

        Assert.True(result.IsSuccess);
        Assert.Null(result.Value);
    }

    [Fact]
    public async Task ClaimNextTask_returns_assignment_when_task_pending()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-123"));

        var handler = new ClaimNextTaskCommandHandler(_queueStore, _options, _weightApplication, _telemetry, _time);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new ClaimNextTaskCommand("worker-alpha"), context);

        Assert.True(result.IsSuccess);
        Assert.NotNull(result.Value);
        Assert.Equal("task-123", result.Value!.TaskId);
        Assert.Equal("http://localhost/weights/1", result.Value.WeightUrl);
        Assert.Equal(4096, result.Value.TokensPerTask);
    }

    // ── SubmitHeartbeatCommand ──────────────────────────────────────

    [Fact]
    public async Task SubmitHeartbeat_touches_worker_and_returns_server_time()
    {
        _workerStore.Upsert(new WorkerRecord(
            WorkerId: "worker-alpha",
            Name: "alpha",
            CpuThreads: 4,
            TokensPerSecond: 1000d,
            RecommendedTokensPerTask: 150_016L,
            ProcessArchitecture: "X64",
            OsDescription: "TestOS",
            RegisteredAtUtc: _time.GetUtcNow(),
            LastHeartbeatUtc: _time.GetUtcNow(),
            State: WorkerState.Active));

        var handler = new SubmitHeartbeatCommandHandler(_workerStore, _time);
        var command = new SubmitHeartbeatCommand(
            "worker-alpha",
            new HeartbeatRequest("worker-alpha", "idle", null, 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsSuccess);
        Assert.False(result.Value!.ShouldDrain);
    }

    [Fact]
    public async Task SubmitHeartbeat_returns_unregistered_failure_for_unknown_worker()
    {
        var handler = new SubmitHeartbeatCommandHandler(_workerStore, _time);
        var command = new SubmitHeartbeatCommand(
            "ghost-worker",
            new HeartbeatRequest("ghost-worker", "idle", null, 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
        Assert.Equal(SubmitHeartbeatCommandHandler.UnregisteredFailureCode, result.Error);
    }

    // ── SubmitGradientCommand ───────────────────────────────────────

    private SubmitGradientCommandHandler BuildGradientHandler() =>
        new(_queueStore, _weightApplication, _telemetry, NullLogger<SubmitGradientCommandHandler>.Instance);

    [Fact]
    public async Task SubmitGradient_marks_task_done_on_happy_path()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-grad"));
        var claimed = _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));
        Assert.NotNull(claimed);

        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-grad",
                WorkerId: "worker-alpha",
                BaseWeightVersion: 1,
                TokensSeen: 4096,
                LossAfter: 1.0,
                GradientFormat: "stub-noop",
                GradientPayload: Array.Empty<byte>(),
                WallClockMs: 500));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsSuccess);
        Assert.Equal(1, _queueStore.CountByState(WorkTaskState.Done));
    }

    [Fact]
    public async Task SubmitGradient_rejects_worker_mismatch()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-mismatch"));
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));

        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-mismatch",
                WorkerId: "imposter",
                BaseWeightVersion: 1,
                TokensSeen: 0,
                LossAfter: 0,
                GradientFormat: "stub-noop",
                GradientPayload: Array.Empty<byte>(),
                WallClockMs: 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
        Assert.Equal(SubmitGradientCommandHandler.WorkerMismatchCode, result.Error);
    }

    [Fact]
    public async Task SubmitGradient_applies_real_int8_gradient_and_bumps_weight_version()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-real-grad"));
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));

        var gradient = new float[_weightApplication.Dimension];
        for (var i = 0; i < gradient.Length; i++)
        {
            gradient[i] = 0.5f;
        }
        var residual = new float[gradient.Length];
        var payload = Int8GradientCodec.Encode(gradient, residual);
        var versionBefore = _weightApplication.CurrentVersion;

        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-real-grad",
                WorkerId: "worker-alpha",
                BaseWeightVersion: versionBefore,
                TokensSeen: 4096,
                LossAfter: 1.0,
                GradientFormat: Int8GradientCodec.FormatId,
                GradientPayload: payload,
                WallClockMs: 500));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsSuccess);
        Assert.Equal(versionBefore + 1, result.Value!.NewWeightVersion);
        Assert.Equal(0, result.Value.Staleness);
        Assert.True(result.Value.EffectiveLearningRate > 0f);
        Assert.Equal(1, _queueStore.CountByState(WorkTaskState.Done));
        Assert.Equal(versionBefore + 1, _weightApplication.CurrentVersion);
    }

    [Fact]
    public async Task SubmitGradient_rejects_stale_gradient_beyond_max_staleness()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-stale"));
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));

        // Push the weight version forward so the upcoming submission
        // is six versions behind, exceeding the MaxStalenessSteps=5
        // configured in the fixture.
        var gradient = new float[_weightApplication.Dimension];
        gradient[0] = 1f;
        for (var i = 0; i < 6; i++)
        {
            var r = new float[gradient.Length];
            _weightApplication.Apply(_weightApplication.CurrentVersion, Int8GradientCodec.Decode(Int8GradientCodec.Encode(gradient, r)));
        }

        var residual = new float[gradient.Length];
        var payload = Int8GradientCodec.Encode(gradient, residual);
        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-stale",
                WorkerId: "worker-alpha",
                BaseWeightVersion: 1,
                TokensSeen: 0,
                LossAfter: 0,
                GradientFormat: Int8GradientCodec.FormatId,
                GradientPayload: payload,
                WallClockMs: 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
        Assert.Contains(SubmitGradientCommandHandler.StaleGradientCode, result.Error);
    }

    [Fact]
    public async Task SubmitGradient_rejects_malformed_payload()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-bad-payload"));
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));

        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-bad-payload",
                WorkerId: "worker-alpha",
                BaseWeightVersion: 1,
                TokensSeen: 0,
                LossAfter: 0,
                GradientFormat: Int8GradientCodec.FormatId,
                GradientPayload: new byte[] { 1, 2, 3, 4 }, // nowhere near a valid blob
                WallClockMs: 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
        Assert.Contains(SubmitGradientCommandHandler.InvalidPayloadCode, result.Error);
    }

    [Fact]
    public async Task SubmitGradient_rejects_unknown_format()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-unknown-fmt"));
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));

        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-unknown-fmt",
                WorkerId: "worker-alpha",
                BaseWeightVersion: 1,
                TokensSeen: 0,
                LossAfter: 0,
                GradientFormat: "mystery-codec-v99",
                GradientPayload: new byte[] { 0 },
                WallClockMs: 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
        Assert.Contains(SubmitGradientCommandHandler.InvalidPayloadCode, result.Error);
    }

    [Fact]
    public async Task SubmitGradient_rejects_unassigned_task()
    {
        // Task exists but was never claimed by worker-alpha.
        _queueStore.EnqueuePending(NewPendingTask("task-ghost"));

        var handler = BuildGradientHandler();
        var command = new SubmitGradientCommand(
            "worker-alpha",
            new GradientSubmission(
                TaskId: "task-ghost",
                WorkerId: "worker-alpha",
                BaseWeightVersion: 1,
                TokensSeen: 0,
                LossAfter: 0,
                GradientFormat: "stub-noop",
                GradientPayload: Array.Empty<byte>(),
                WallClockMs: 0));

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsFailure);
        Assert.Equal(SubmitGradientCommandHandler.TaskNotAssignedCode, result.Error);
    }

    // ── EnqueueTasksCommand ─────────────────────────────────────────

    private EnqueueTasksCommandHandler BuildEnqueueHandler() =>
        new(_queueStore, _time, NullLogger<EnqueueTasksCommandHandler>.Instance);

    [Fact]
    public async Task EnqueueTasks_inserts_requested_count_with_monotonic_offsets()
    {
        var handler = BuildEnqueueHandler();
        var command = new EnqueueTasksCommand(
            ShardId: "shard-bulk",
            ShardStartOffset: 0,
            ShardStride: 4096,
            TokensPerTask: 4096,
            KLocalSteps: 4,
            HyperparametersJson: "{\"lr\":1e-3}",
            WeightVersion: 1,
            Count: 3);

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsSuccess);
        Assert.Equal(3, result.Value!.Inserted);
        Assert.NotEqual(result.Value.FirstTaskId, result.Value.LastTaskId);
        Assert.Equal(3, _queueStore.CountByState(WorkTaskState.Pending));
    }

    [Fact]
    public async Task EnqueueTasks_rejects_non_positive_count()
    {
        var handler = BuildEnqueueHandler();

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new EnqueueTasksCommand("shard-x", 0, 0, 4096, 4, "{}", 1, 0),
            context);

        Assert.True(result.IsFailure);
    }

    [Fact]
    public async Task EnqueueTasks_rejects_empty_shard_id()
    {
        var handler = BuildEnqueueHandler();

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new EnqueueTasksCommand("", 0, 0, 4096, 4, "{}", 1, 1),
            context);

        Assert.True(result.IsFailure);
    }

    [Fact]
    public async Task EnqueueTasks_persists_requested_tokens_per_task_on_every_record()
    {
        // Verifies the seed CLI / admin enqueue path round-trips the
        // caller-specified TokensPerTask into each stored WorkTaskRecord
        // instead of collapsing to a stub value. This is the invariant
        // that keeps queued task sizes aligned with the worker's
        // capability-pass target (TaskSizingCalculator output).
        const long LargeTokenBudget = 262_144L;
        const int EnqueueCount = 4;

        var handler = BuildEnqueueHandler();
        var command = new EnqueueTasksCommand(
            ShardId: "shard-sized",
            ShardStartOffset: 0,
            ShardStride: LargeTokenBudget,
            TokensPerTask: LargeTokenBudget,
            KLocalSteps: 4,
            HyperparametersJson: "{}",
            WeightVersion: 1,
            Count: EnqueueCount);

        using var context = new CallContext();
        var result = await handler.HandleAsync(command, context);

        Assert.True(result.IsSuccess);
        Assert.Equal(EnqueueCount, result.Value!.Inserted);

        var first = _queueStore.GetById(result.Value.FirstTaskId);
        var last = _queueStore.GetById(result.Value.LastTaskId);
        Assert.NotNull(first);
        Assert.NotNull(last);
        Assert.Equal(LargeTokenBudget, first!.TokensPerTask);
        Assert.Equal(LargeTokenBudget, last!.TokensPerTask);
        Assert.Equal(LargeTokenBudget, first.ShardLength);
        Assert.Equal(0L, first.ShardOffset);
        Assert.Equal((long)(EnqueueCount - 1) * LargeTokenBudget, last.ShardOffset);
    }

    // ── GetTaskQueueSnapshotQuery ───────────────────────────────────

    [Fact]
    public async Task GetTaskQueueSnapshot_reflects_current_state()
    {
        _queueStore.EnqueuePending(NewPendingTask("snap-1"));
        _queueStore.EnqueuePending(NewPendingTask("snap-2"));
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));
        _queueStore.EnqueuePending(NewPendingTask("snap-3"));

        var handler = new GetTaskQueueSnapshotQueryHandler(_queueStore);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new GetTaskQueueSnapshotQuery(), context);

        Assert.True(result.IsSuccess);
        Assert.Equal(2, result.Value!.Pending);
        Assert.Equal(1, result.Value.Assigned);
        Assert.Equal(0, result.Value.Done);
        Assert.Equal(0, result.Value.Failed);
    }

    // ── GetDashboardSnapshotQuery ───────────────────────────────────

    [Fact]
    public async Task GetDashboardSnapshot_reports_progress_and_eta_from_live_window()
    {
        _workerStore.Upsert(new WorkerRecord(
            WorkerId: "worker-alpha",
            Name: "alpha",
            CpuThreads: 4,
            TokensPerSecond: 1000d,
            RecommendedTokensPerTask: 150_016L,
            ProcessArchitecture: "X64",
            OsDescription: "TestOS",
            RegisteredAtUtc: _time.GetUtcNow(),
            LastHeartbeatUtc: _time.GetUtcNow(),
            State: WorkerState.Active));

        // Queue: 3 pending + 1 assigned + 2 done.
        _queueStore.EnqueuePending(NewPendingTask("d-1"));
        _queueStore.EnqueuePending(NewPendingTask("d-2"));
        _queueStore.EnqueuePending(NewPendingTask("d-3"));
        _queueStore.EnqueuePending(NewPendingTask("d-4"));
        var done1 = _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));
        _queueStore.MarkCompleted(done1!.TaskId, "worker-alpha");
        var done2 = _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));
        _queueStore.MarkCompleted(done2!.TaskId, "worker-alpha");
        _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10)); // leaves one Assigned

        // Two telemetry events inside the 1-min live window so the
        // ETA has a real rate to divide by.
        _telemetry.RecordAccepted("worker-alpha", "d-1", 4096, 500, 0, 0.1f, 2, 1.0d);
        _telemetry.RecordAccepted("worker-alpha", "d-2", 4096, 500, 0, 0.1f, 3, 0.9d);

        var handler = new GetDashboardSnapshotQueryHandler(
            _queueStore,
            _workerStore,
            _telemetry,
            _weightApplication,
            _time);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new GetDashboardSnapshotQuery(), context);

        Assert.True(result.IsSuccess);
        var snap = result.Value!;

        Assert.Equal(4, snap.Progress.TotalTasks);
        Assert.Equal(2, snap.Progress.CompletedTasks);
        Assert.Equal(2, snap.Progress.RemainingTasks);
        Assert.InRange(snap.Progress.PercentComplete, 0.49d, 0.51d);

        // tasks/s over the 60 s live window = 2 / 60.
        Assert.InRange(snap.Progress.TasksPerSecondLive, 0.032d, 0.034d);
        Assert.NotNull(snap.Progress.EtaSeconds);
        Assert.NotNull(snap.Progress.EtaUtc);

        // Live window aggregate surfaces on the snapshot.
        Assert.Equal(2, snap.LiveGlobalTelemetry.TasksCompleted);
    }

    [Fact]
    public async Task GetDashboardSnapshot_populates_per_worker_live_columns()
    {
        _workerStore.Upsert(new WorkerRecord(
            WorkerId: "worker-alpha",
            Name: "alpha",
            CpuThreads: 4,
            TokensPerSecond: 1000d,
            RecommendedTokensPerTask: 150_016L,
            ProcessArchitecture: "X64",
            OsDescription: "TestOS",
            RegisteredAtUtc: _time.GetUtcNow(),
            LastHeartbeatUtc: _time.GetUtcNow(),
            State: WorkerState.Active));

        _queueStore.EnqueuePending(NewPendingTask("live-1"));
        var claimed = _queueStore.TryClaimNextPending("worker-alpha", TimeSpan.FromMinutes(10));
        Assert.NotNull(claimed);

        _telemetry.RecordAccepted("worker-alpha", "live-1", 4096, 500, 0, 0.1f, 2, 1.0d);

        var handler = new GetDashboardSnapshotQueryHandler(
            _queueStore,
            _workerStore,
            _telemetry,
            _weightApplication,
            _time);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new GetDashboardSnapshotQuery(), context);

        Assert.True(result.IsSuccess);
        var row = Assert.Single(result.Value!.WorkerRows);
        Assert.Equal("worker-alpha", row.ClientId);
        Assert.Equal(1, row.LiveTasksCompleted);
        Assert.Equal(4096L, row.LiveTokensSeen);
        Assert.True(row.LiveTokensPerSecond > 0d);
        // Because the task is Assigned and live, CurrentTaskId should surface.
        Assert.Equal("live-1", row.CurrentTaskId);
        Assert.NotNull(row.CurrentTaskStartedUtc);
        Assert.NotNull(row.SecondsOnCurrentTask);
    }

    [Fact]
    public async Task EnqueueTasks_defaults_stride_to_tokens_per_task()
    {
        var handler = BuildEnqueueHandler();

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new EnqueueTasksCommand(
                ShardId: "shard-default-stride",
                ShardStartOffset: 1000,
                ShardStride: 0,
                TokensPerTask: 512,
                KLocalSteps: 4,
                HyperparametersJson: "",
                WeightVersion: 1,
                Count: 2),
            context);

        Assert.True(result.IsSuccess);
        Assert.Equal(2, result.Value!.Inserted);
    }

    [Fact]
    public void GetGlobalMeasuredTokensPerSecond_returns_null_when_empty()
    {
        Assert.Null(_telemetry.GetGlobalMeasuredTokensPerSecond());
    }

    [Fact]
    public void GetGlobalMeasuredTokensPerSecond_uses_sum_across_workers()
    {
        _telemetry.RecordAccepted(
            clientId: "w1", taskId: "t1", tokensSeen: 1000,
            wallClockMs: 2000, staleness: 0, effectiveLr: 0.1f,
            newVersion: 1, lossAfter: 2.0);
        _telemetry.RecordAccepted(
            clientId: "w2", taskId: "t2", tokensSeen: 3000,
            wallClockMs: 2000, staleness: 0, effectiveLr: 0.1f,
            newVersion: 2, lossAfter: 2.0);

        // 4000 tokens / 4.0s = 1000 tok/s
        var tps = _telemetry.GetGlobalMeasuredTokensPerSecond();
        Assert.NotNull(tps);
        Assert.Equal(1000.0, tps!.Value, precision: 3);
    }

    [Fact]
    public void GetGlobalMeasuredTokensPerSecond_ignores_events_outside_window()
    {
        _telemetry.RecordAccepted(
            clientId: "w1", taskId: "t-old", tokensSeen: 999_999,
            wallClockMs: 10, staleness: 0, effectiveLr: 0.1f,
            newVersion: 1, lossAfter: 2.0);

        _time.Advance(TimeSpan.FromHours(2));

        _telemetry.RecordAccepted(
            clientId: "w1", taskId: "t-new", tokensSeen: 1000,
            wallClockMs: 2000, staleness: 0, effectiveLr: 0.1f,
            newVersion: 2, lossAfter: 2.0);

        // Only the recent 1000 tok / 2s = 500 should count; the
        // 2-hour-old synthetic event sits outside the 30-min window.
        var tps = _telemetry.GetGlobalMeasuredTokensPerSecond();
        Assert.NotNull(tps);
        Assert.Equal(500.0, tps!.Value, precision: 3);
    }
}
#endif
