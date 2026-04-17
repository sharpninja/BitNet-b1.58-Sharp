using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// End-to-end integration test exercising the full worker lifecycle
/// against the real coordinator Web host via
/// <see cref="WebApplicationFactory{TEntryPoint}"/>. Walks through
/// /register → /heartbeat → /work → /gradient and asserts the SQLite
/// task state is Done after the round trip. Every call carries the
/// shared <c>X-Api-Key</c> + worker-specific <c>X-Worker-Id</c>
/// headers — no OAuth, no JWT.
/// </summary>
public sealed class CoordinatorEndToEndTests : IClassFixture<CoordinatorEndToEndTests.E2EFactory>, IDisposable
{
    public sealed class E2EFactory : WebApplicationFactory<CoordinatorHostMarker>
    {
        public string DatabasePath { get; } =
            Path.Combine(Path.GetTempPath(), $"bitnet-coord-e2e-{Guid.NewGuid():N}.db");

        public string WeightsDirectory => Path.Combine(Path.GetDirectoryName(DatabasePath) ?? ".", "weights");

        public const string WorkerId = "worker-e2e";
        public const string ApiKey = "e2e-shared-api-key";
        public const string AdminUsername = "admin-e2e";
        public const string AdminPassword = "admin-e2e-password";

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            builder.ConfigureAppConfiguration((_, config) =>
            {
                var testSettings = new Dictionary<string, string?>
                {
                    ["Coordinator:DatabasePath"] = DatabasePath,
                    ["Coordinator:BaseUrl"] = "http://localhost",
                    ["Coordinator:HeartbeatIntervalSeconds"] = "30",
                    ["Coordinator:StaleWorkerThresholdSeconds"] = "300",
                    ["Coordinator:TargetTaskDurationSeconds"] = "600",
                    ["Coordinator:FullStepEfficiency"] = "0.25",
                    ["Coordinator:InitialWeightVersion"] = "1",
                    ["Coordinator:Admin:Username"] = AdminUsername,
                    ["Coordinator:Admin:Password"] = AdminPassword,
                    ["Coordinator:WorkerApiKey"] = ApiKey
                };
                config.AddInMemoryCollection(testSettings);
            });
        }

        public void CleanupDatabase()
        {
            TryDelete(DatabasePath);
            TryDelete(DatabasePath + "-wal");
            TryDelete(DatabasePath + "-shm");
            if (Directory.Exists(WeightsDirectory))
            {
                try { Directory.Delete(WeightsDirectory, recursive: true); } catch { }
            }
        }

        private static void TryDelete(string path)
        {
            try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
        }
    }

    private readonly E2EFactory _factory;
    private readonly HttpClient _client;

    public CoordinatorEndToEndTests(E2EFactory factory)
    {
        _factory = factory;

        _client = _factory.CreateClient(new WebApplicationFactoryClientOptions
        {
            AllowAutoRedirect = false,
            BaseAddress = new Uri("http://localhost/")
        });
    }

    public void Dispose()
    {
        _client.Dispose();
    }

    private HttpRequestMessage Authed(HttpMethod method, string path, HttpContent? body = null)
    {
        var req = new HttpRequestMessage(method, path);
        req.Headers.Add("X-Api-Key", E2EFactory.ApiKey);
        req.Headers.Add("X-Worker-Id", E2EFactory.WorkerId);
        if (body is not null)
        {
            req.Content = body;
        }
        return req;
    }

    [Fact]
    public async Task Full_worker_lifecycle_registers_claims_and_completes_a_task()
    {
        // 1. POST /register — worker comes online using the shared key.
        var registrationRequest = new WorkerRegistrationRequest(
            WorkerName: "e2e-worker",
            EnrollmentKey: string.Empty,
            ProcessArchitecture: "X64",
            OsDescription: "TestOS",
            Capability: new WorkerCapabilityDto(
                TokensPerSecond: 1000d,
                CpuThreads: 4,
                CalibrationDurationMs: 1234,
                BenchmarkId: "Int8TernaryMatMul",
                MeasuredAt: DateTimeOffset.UtcNow));

        using (var registerResponse = await _client.SendAsync(
            Authed(HttpMethod.Post, "/register", JsonContent.Create(registrationRequest))))
        {
            if (!registerResponse.IsSuccessStatusCode)
            {
                var diag = await registerResponse.Content.ReadAsStringAsync();
                var auth = registerResponse.Headers.WwwAuthenticate.ToString();
                throw new InvalidOperationException(
                    $"/register failed with {(int)registerResponse.StatusCode}. body={diag}; www-authenticate={auth}");
            }

            var body = await registerResponse.Content.ReadFromJsonAsync<WorkerRegistrationResponse>();
            Assert.NotNull(body);
            Assert.Equal(E2EFactory.WorkerId, body!.WorkerId);
            Assert.Equal(150_016L, body.RecommendedTokensPerTask);
        }

        // 2. POST /heartbeat — server accepts the keep-alive.
        var heartbeat = new HeartbeatRequest(
            WorkerId: E2EFactory.WorkerId,
            Status: "idle",
            CurrentTaskId: null,
            TokensSeenSinceLastHeartbeat: 0);
        using (var heartbeatResponse = await _client.SendAsync(
            Authed(HttpMethod.Post, "/heartbeat", JsonContent.Create(heartbeat))))
        {
            heartbeatResponse.EnsureSuccessStatusCode();
        }

        // 3. Seed one pending task directly into the queue store so
        //    the /work endpoint has something to hand back. This is a
        //    back-door into the coordinator's SQLite file via the
        //    same singleton the host uses.
        var queueStore = _factory.Services.GetRequiredService<SqliteWorkQueueStore>();
        queueStore.EnqueuePending(new WorkTaskRecord(
            TaskId: "task-e2e-001",
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
            CreatedAtUtc: DateTimeOffset.UtcNow,
            CompletedAtUtc: null));

        // 4. GET /work — expect the task back.
        WorkTaskAssignment? assignment;
        using (var workResponse = await _client.SendAsync(
            Authed(HttpMethod.Get, "/work")))
        {
            Assert.Equal(HttpStatusCode.OK, workResponse.StatusCode);
            assignment = await workResponse.Content.ReadFromJsonAsync<WorkTaskAssignment>();
            Assert.NotNull(assignment);
            Assert.Equal("task-e2e-001", assignment!.TaskId);
        }

        // 5. POST /gradient — report completion.
        var submission = new GradientSubmission(
            TaskId: assignment.TaskId,
            WorkerId: E2EFactory.WorkerId,
            BaseWeightVersion: assignment.WeightVersion,
            TokensSeen: assignment.TokensPerTask,
            LossAfter: 0.5,
            GradientFormat: "stub-noop",
            GradientPayload: Array.Empty<byte>(),
            WallClockMs: 500);
        using (var gradientResponse = await _client.SendAsync(
            Authed(HttpMethod.Post, "/gradient", JsonContent.Create(submission))))
        {
            gradientResponse.EnsureSuccessStatusCode();
        }

        // 6. Verify final state: task is Done.
        Assert.Equal(1, queueStore.CountByState(WorkTaskState.Done));
        Assert.Equal(0, queueStore.CountByState(WorkTaskState.Assigned));
        Assert.Equal(0, queueStore.CountByState(WorkTaskState.Pending));
    }
}
