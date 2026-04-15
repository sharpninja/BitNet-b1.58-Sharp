#if NET10_0_OR_GREATER
using System;
using System.IO;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Cqrs.Commands;
using BitNetSharp.Distributed.Coordinator.Cqrs.Queries;
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
    private readonly FakeTimeProvider _time;
    private readonly SqliteWorkerRegistryStore _workerStore;
    private readonly SqliteWorkQueueStore _queueStore;
    private readonly SqliteClientRevocationStore _revocations;
    private readonly WorkerClientRegistry _registry;
    private readonly IOptionsMonitor<CoordinatorOptions> _options;

    public CqrsHandlerTests()
    {
        _databasePath = Path.Combine(Path.GetTempPath(), $"bitnet-cqrs-{Guid.NewGuid():N}.db");
        _time = new FakeTimeProvider(new DateTimeOffset(2026, 4, 15, 19, 0, 0, TimeSpan.Zero));
        var connectionString = $"Data Source={_databasePath}";
        _workerStore = new SqliteWorkerRegistryStore(connectionString, _time);
        _queueStore = new SqliteWorkQueueStore(connectionString, _time);
        _revocations = new SqliteClientRevocationStore(connectionString, _time);

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
            BaseUrl = "http://localhost"
        });
    }

    public void Dispose()
    {
        _workerStore.Dispose();
        _queueStore.Dispose();
        _revocations.Dispose();
        TryDelete(_databasePath);
        TryDelete(_databasePath + "-wal");
        TryDelete(_databasePath + "-shm");
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
        new(_queueStore, NullLogger<SubmitGradientCommandHandler>.Instance);

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
}
#endif
