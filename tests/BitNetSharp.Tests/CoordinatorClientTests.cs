using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Worker;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for <see cref="CoordinatorClient"/> using a
/// recording stub HttpMessageHandler so the register + heartbeat +
/// work + gradient endpoints can be exercised without spinning up a
/// real TestServer. Verifies the shared X-Api-Key / X-Worker-Id
/// headers ride on every outbound request.
/// </summary>
public sealed class CoordinatorClientTests
{
    private static WorkerConfig SampleConfig() =>
        new(
            CoordinatorUrl: new Uri("https://coord.example.test/"),
            ApiKey: "shared-secret-key",
            WorkerId: "worker-alpha",
            WorkerName: "test-worker",
            CpuThreads: 4,
            HeartbeatInterval: TimeSpan.FromSeconds(30),
            ShutdownGrace: TimeSpan.FromSeconds(30),
            HealthBeaconPath: "/tmp/test-beacon",
            LogLevel: "info",
            ModelPreset: "small");

    private sealed class StubHandler : HttpMessageHandler
    {
        public List<HttpRequestMessage> Requests { get; } = new();
        public Queue<HttpResponseMessage> Responses { get; } = new();

        public void Enqueue(HttpStatusCode status, string? body = null)
        {
            var message = new HttpResponseMessage(status);
            if (body is not null)
            {
                message.Content = new StringContent(body, Encoding.UTF8, "application/json");
            }
            Responses.Enqueue(message);
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            if (Responses.Count == 0)
            {
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.InternalServerError));
            }

            return Task.FromResult(Responses.Dequeue());
        }
    }

    private static CoordinatorClient CreateClient(StubHandler handler, WorkerConfig config)
    {
        var httpClient = new HttpClient(handler)
        {
            BaseAddress = config.CoordinatorUrl
        };
        return new CoordinatorClient(config, httpClient, ownsHttpClient: true);
    }

    private static void AssertAuthHeaders(HttpRequestMessage request, WorkerConfig config)
    {
        Assert.True(request.Headers.Contains(CoordinatorClient.ApiKeyHeader),
            $"Expected {CoordinatorClient.ApiKeyHeader} header on outgoing request");
        Assert.Equal(config.ApiKey,
            request.Headers.GetValues(CoordinatorClient.ApiKeyHeader).Single());

        Assert.True(request.Headers.Contains(CoordinatorClient.WorkerIdHeader),
            $"Expected {CoordinatorClient.WorkerIdHeader} header on outgoing request");
        Assert.Equal(config.WorkerId,
            request.Headers.GetValues(CoordinatorClient.WorkerIdHeader).Single());
    }

    [Fact]
    public async Task RegisterAsync_sends_api_key_and_worker_id_headers_and_deserializes_response()
    {
        var config = SampleConfig();
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        {
          "workerId": "worker-alpha",
          "bearerToken": "",
          "initialWeightVersion": 42,
          "recommendedTokensPerTask": 150016,
          "heartbeatIntervalSeconds": 30,
          "serverTime": "2026-04-15T18:00:00+00:00"
        }
        """);

        using var client = CreateClient(handler, config);

        var request = new WorkerRegistrationRequest(
            WorkerName: "alpha",
            EnrollmentKey: string.Empty,
            ProcessArchitecture: "X64",
            OsDescription: "TestOS",
            Capability: new WorkerCapabilityDto(
                TokensPerSecond: 1000d,
                CpuThreads: 4,
                CalibrationDurationMs: 1234,
                BenchmarkId: "Int8TernaryMatMul",
                MeasuredAt: DateTimeOffset.UtcNow));

        var response = await client.RegisterAsync(request);

        Assert.Equal("worker-alpha", response.WorkerId);
        Assert.Equal(42, response.InitialWeightVersion);
        Assert.Equal(150016, response.RecommendedTokensPerTask);

        Assert.Single(handler.Requests);
        var sent = handler.Requests[0];
        Assert.EndsWith("/register", sent.RequestUri!.AbsolutePath);
        Assert.Equal(HttpMethod.Post, sent.Method);
        AssertAuthHeaders(sent, config);
        // Ensure the old OAuth Bearer scheme is not being attached.
        Assert.Null(sent.Headers.Authorization);
    }

    [Fact]
    public async Task TryClaimWorkAsync_returns_null_on_204_and_sends_auth_headers()
    {
        var config = SampleConfig();
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.NoContent);

        using var client = CreateClient(handler, config);

        var work = await client.TryClaimWorkAsync();

        Assert.Null(work);
        Assert.Single(handler.Requests);
        var sent = handler.Requests[0];
        Assert.EndsWith("/work", sent.RequestUri!.AbsolutePath);
        AssertAuthHeaders(sent, config);
    }

    [Fact]
    public async Task SendHeartbeatAsync_returns_null_on_410_gone_and_sends_auth_headers()
    {
        var config = SampleConfig();
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.Gone);

        using var client = CreateClient(handler, config);

        var response = await client.SendHeartbeatAsync(new HeartbeatRequest(
            WorkerId: config.WorkerId,
            Status: "idle",
            CurrentTaskId: null,
            TokensSeenSinceLastHeartbeat: 0));

        Assert.Null(response);
        Assert.Single(handler.Requests);
        var sent = handler.Requests[0];
        Assert.EndsWith("/heartbeat", sent.RequestUri!.AbsolutePath);
        AssertAuthHeaders(sent, config);
    }

    [Fact]
    public async Task SubmitGradientAsync_returns_true_on_200()
    {
        var config = SampleConfig();
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "accepted": true, "task_id": "task-1", "worker_id": "worker-alpha" }
        """);

        using var client = CreateClient(handler, config);
        var submission = new GradientSubmission(
            TaskId: "task-1",
            WorkerId: config.WorkerId,
            BaseWeightVersion: 42,
            TokensSeen: 4096,
            LossAfter: 1.23,
            GradientFormat: "int8-ef",
            GradientPayload: new byte[] { 1, 2, 3 },
            WallClockMs: 1000);

        var accepted = await client.SubmitGradientAsync(submission);
        Assert.True(accepted);
        AssertAuthHeaders(handler.Requests[0], config);
    }

    [Fact]
    public async Task SubmitGradientAsync_returns_false_on_403()
    {
        var config = SampleConfig();
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.Forbidden);

        using var client = CreateClient(handler, config);
        var accepted = await client.SubmitGradientAsync(new GradientSubmission(
            TaskId: "task-1",
            WorkerId: config.WorkerId,
            BaseWeightVersion: 42,
            TokensSeen: 0,
            LossAfter: 0,
            GradientFormat: "int8-ef",
            GradientPayload: Array.Empty<byte>(),
            WallClockMs: 0));
        Assert.False(accepted);
    }

    [Fact]
    public async Task SubmitGradientAsync_returns_false_on_409()
    {
        var config = SampleConfig();
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.Conflict);

        using var client = CreateClient(handler, config);
        var accepted = await client.SubmitGradientAsync(new GradientSubmission(
            TaskId: "task-1",
            WorkerId: config.WorkerId,
            BaseWeightVersion: 42,
            TokensSeen: 0,
            LossAfter: 0,
            GradientFormat: "int8-ef",
            GradientPayload: Array.Empty<byte>(),
            WallClockMs: 0));
        Assert.False(accepted);
    }
}
