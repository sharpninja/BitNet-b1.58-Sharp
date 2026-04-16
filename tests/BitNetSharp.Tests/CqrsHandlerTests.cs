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
using BitNetSharp.Distributed.Coordinator.Identity;
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
    private readonly SqliteClientRevocationStore _revocations;
    private readonly SqliteTelemetryStore _telemetry;
    private readonly FileSystemWeightStore _weightStore;
    private readonly WeightApplicationService _weightApplication;
    private readonly WorkerClientRegistry _registry;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;

    public CqrsHandlerTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-cqrs-{Guid.NewGuid():N}.db");
        _weightsDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-cqrs-weights-{Guid.NewGuid():N}");
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 19, 0, 0, TimeSpan.Zero));
        var connectionString = $"Data Source={_databasePath}";
        _workerStore = new SqliteWorkerRegistryStore(connectionString, _time);
        _queueStore = new SqliteWorkQueueStore(connectionString, _time);
        _revocations = new SqliteClientRevocationStore(connectionString, _time);
        _telemetry = new SqliteTelemetryStore(connectionString, _time);
        _weightStore = new FileSystemWeightStore(_weightsDirectory);

        _registry = new WorkerClientRegistry();
        _registry.Seed(new[]
        {
            new WorkerClientOptions
            {
                ClientId = "worker-alpha",
                ClientSecret = "alpha-secret",
                DisplayName = "Alpha"
            }
        });

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
        _revocations.Dispose();
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
        var handler = new ClaimNextTaskCommandHandler(_queueStore, _options, _time);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new ClaimNextTaskCommand("worker-alpha"), context);

        Assert.True(result.IsSuccess);
        Assert.Null(result.Value);
    }

    [Fact]
    public async Task ClaimNextTask_returns_assignment_when_task_pending()
    {
        _queueStore.EnqueuePending(NewPendingTask("task-123"));

        var handler = new ClaimNextTaskCommandHandler(_queueStore, _options, _time);

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

    // ── GetWorkerClientsQuery ───────────────────────────────────────

    // ── GetWorkerInstallScriptQuery ─────────────────────────────────

    [Fact]
    public async Task GetWorkerInstallScript_renders_bash_for_known_client()
    {
        var handler = new GetWorkerInstallScriptQueryHandler(_registry, _options);

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new GetWorkerInstallScriptQuery("worker-alpha", InstallShell.Bash),
            context);

        Assert.True(result.IsSuccess);
        Assert.EndsWith(".sh", result.Value!.Filename);
        Assert.Contains("BITNET_CLIENT_ID=\"worker-alpha\"", result.Value.Content);
        Assert.Contains("BITNET_CLIENT_SECRET=\"alpha-secret\"", result.Value.Content);
        Assert.Contains("http://localhost", result.Value.Content);
    }

    [Fact]
    public async Task GetWorkerInstallScript_renders_powershell_for_known_client()
    {
        var handler = new GetWorkerInstallScriptQueryHandler(_registry, _options);

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new GetWorkerInstallScriptQuery("worker-alpha", InstallShell.PowerShell),
            context);

        Assert.True(result.IsSuccess);
        Assert.EndsWith(".ps1", result.Value!.Filename);
        Assert.Contains("BITNET_CLIENT_ID         = 'worker-alpha'", result.Value.Content);
        Assert.Contains("BITNET_CLIENT_SECRET     = 'alpha-secret'", result.Value.Content);
    }

    [Fact]
    public async Task GetWorkerInstallScript_fails_for_unknown_client()
    {
        var handler = new GetWorkerInstallScriptQueryHandler(_registry, _options);

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new GetWorkerInstallScriptQuery("ghost-client", InstallShell.Bash),
            context);

        Assert.True(result.IsFailure);
        Assert.Contains("Unknown worker client", result.Error);
    }

    [Fact]
    public async Task GetWorkerClients_returns_registry_entries_with_revocation_status()
    {
        _revocations.Revoke("worker-alpha");

        var handler = new GetWorkerClientsQueryHandler(_registry, _revocations);

        using var context = new CallContext();
        var result = await handler.HandleAsync(new GetWorkerClientsQuery(), context);

        Assert.True(result.IsSuccess);
        Assert.Single(result.Value!);
        var view = result.Value![0];
        Assert.Equal("worker-alpha", view.ClientId);
        Assert.Equal("alpha-secret", view.ClientSecret);
        Assert.NotNull(view.RevokedAtUtc);
    }

    // ── RotateClientSecretCommand ───────────────────────────────────

    [Fact]
    public async Task RotateClientSecret_generates_new_secret_and_revokes()
    {
        var handler = new RotateClientSecretCommandHandler(
            _registry,
            _revocations,
            NullLogger<RotateClientSecretCommandHandler>.Instance);

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new RotateClientSecretCommand("worker-alpha"),
            context);

        Assert.True(result.IsSuccess);
        Assert.Equal("worker-alpha", result.Value!.ClientId);
        Assert.NotEqual("alpha-secret", result.Value.NewSecret);
        Assert.NotNull(_revocations.GetRevokedAt("worker-alpha"));
    }

    [Fact]
    public async Task RotateClientSecret_fails_for_unknown_client()
    {
        var handler = new RotateClientSecretCommandHandler(
            _registry,
            _revocations,
            NullLogger<RotateClientSecretCommandHandler>.Instance);

        using var context = new CallContext();
        var result = await handler.HandleAsync(
            new RotateClientSecretCommand("ghost-client"),
            context);

        Assert.True(result.IsFailure);
        Assert.NotNull(result.Error);
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
}
#endif
