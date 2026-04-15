using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Worker;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for <see cref="CoordinatorClient"/> using a
/// recording stub HttpMessageHandler so the token flow + register +
/// heartbeat + work endpoints can be exercised without spinning up
/// a real TestServer.
/// </summary>
public sealed class CoordinatorClientTests
{
    private static WorkerConfig SampleConfig() =>
        new(
            CoordinatorUrl: new Uri("https://coord.example.test/"),
            ClientId: "worker-alpha",
            ClientSecret: "alpha-secret",
            WorkerName: "test-worker",
            CpuThreads: 4,
            HeartbeatInterval: TimeSpan.FromSeconds(30),
            ShutdownGrace: TimeSpan.FromSeconds(30),
            HealthBeaconPath: "/tmp/test-beacon",
            LogLevel: "info");

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

    [Fact]
    public async Task GetAccessTokenAsync_POSTs_client_credentials_and_returns_token()
    {
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "tok-abc", "token_type": "Bearer", "expires_in": 3600, "scope": "bitnet-worker" }
        """);

        using var client = CreateClient(handler, SampleConfig());

        var token = await client.GetAccessTokenAsync();

        Assert.Equal("tok-abc", token);
        Assert.Single(handler.Requests);
        var req = handler.Requests[0];
        Assert.Equal(HttpMethod.Post, req.Method);
        Assert.EndsWith("/connect/token", req.RequestUri!.AbsolutePath);
        Assert.Equal("application/x-www-form-urlencoded", req.Content!.Headers.ContentType!.MediaType);
    }

    [Fact]
    public async Task GetAccessTokenAsync_is_cached_across_calls()
    {
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "tok-cached", "token_type": "Bearer", "expires_in": 3600 }
        """);

        using var client = CreateClient(handler, SampleConfig());

        var first  = await client.GetAccessTokenAsync();
        var second = await client.GetAccessTokenAsync();

        Assert.Equal("tok-cached", first);
        Assert.Equal(first, second);
        Assert.Single(handler.Requests); // second call hit the cache
    }

    [Fact]
    public async Task InvalidateAccessToken_forces_a_refresh_on_the_next_call()
    {
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "tok-one", "token_type": "Bearer", "expires_in": 3600 }
        """);
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "tok-two", "token_type": "Bearer", "expires_in": 3600 }
        """);

        using var client = CreateClient(handler, SampleConfig());

        var first = await client.GetAccessTokenAsync();
        client.InvalidateAccessToken();
        var second = await client.GetAccessTokenAsync();

        Assert.Equal("tok-one", first);
        Assert.Equal("tok-two", second);
        Assert.Equal(2, handler.Requests.Count);
    }

    [Fact]
    public async Task RegisterAsync_attaches_bearer_token_and_deserializes_response()
    {
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "tok-reg", "token_type": "Bearer", "expires_in": 3600 }
        """);
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

        using var client = CreateClient(handler, SampleConfig());

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

        // Second request was /register, must have Bearer auth.
        var registerRequest = handler.Requests[1];
        Assert.EndsWith("/register", registerRequest.RequestUri!.AbsolutePath);
        Assert.Equal("Bearer", registerRequest.Headers.Authorization!.Scheme);
        Assert.Equal("tok-reg", registerRequest.Headers.Authorization.Parameter);
    }

    [Fact]
    public async Task TryClaimWorkAsync_returns_null_on_204()
    {
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "t", "expires_in": 3600 }
        """);
        handler.Enqueue(HttpStatusCode.NoContent);

        using var client = CreateClient(handler, SampleConfig());

        var work = await client.TryClaimWorkAsync();

        Assert.Null(work);
    }

    [Fact]
    public async Task SendHeartbeatAsync_returns_null_on_410_gone()
    {
        var handler = new StubHandler();
        handler.Enqueue(HttpStatusCode.OK, """
        { "access_token": "t", "expires_in": 3600 }
        """);
        handler.Enqueue(HttpStatusCode.Gone);

        using var client = CreateClient(handler, SampleConfig());

        var response = await client.SendHeartbeatAsync(new HeartbeatRequest(
            WorkerId: "worker-alpha",
            Status: "idle",
            CurrentTaskId: null,
            TokensSeenSinceLastHeartbeat: 0));

        Assert.Null(response);
    }
}
