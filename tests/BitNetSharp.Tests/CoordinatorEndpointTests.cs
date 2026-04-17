#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// WebApplicationFactory-backed smoke tests for the coordinator host.
/// Asserts the public endpoints respond, the admin surface correctly
/// triggers the OIDC challenge chain, and the worker surface rejects
/// unauthenticated traffic (and accepts a request bearing the
/// configured shared <c>X-Api-Key</c>).
/// </summary>
public sealed class CoordinatorEndpointTests : IClassFixture<CoordinatorEndpointTests.CoordinatorFactory>, IDisposable
{
    public const string TestApiKey = "test-shared-worker-key";

    public sealed class CoordinatorFactory : WebApplicationFactory<CoordinatorHostMarker>
    {
        public string DatabasePath { get; } =
            Path.Combine(Path.GetTempPath(), $"bitnet-coord-smoke-{Guid.NewGuid():N}.db");

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            builder.ConfigureAppConfiguration((_, config) =>
            {
                var testSettings = new Dictionary<string, string?>
                {
                    ["Coordinator:DatabasePath"] = DatabasePath,
                    ["Coordinator:BaseUrl"] = "http://localhost",
                    ["Coordinator:HeartbeatIntervalSeconds"] = "30",
                    ["Coordinator:StaleWorkerThresholdSeconds"] = "120",
                    ["Coordinator:Admin:Username"] = "admin",
                    ["Coordinator:Admin:Password"] = "super-secret-test",
                    ["Coordinator:WorkerApiKey"] = TestApiKey
                };
                config.AddInMemoryCollection(testSettings);
            });
        }

        public void CleanupDatabase()
        {
            TryDelete(DatabasePath);
            TryDelete(DatabasePath + "-wal");
            TryDelete(DatabasePath + "-shm");
        }

        private static void TryDelete(string path)
        {
            try { if (File.Exists(path)) { File.Delete(path); } } catch { /* best-effort */ }
        }
    }

    private readonly CoordinatorFactory _factory;
    private readonly HttpClient _client;

    public CoordinatorEndpointTests(CoordinatorFactory factory)
    {
        _factory = factory;
        _client = _factory.CreateClient(new WebApplicationFactoryClientOptions
        {
            AllowAutoRedirect = false
        });
    }

    public void Dispose()
    {
        _client.Dispose();
    }

    [Fact]
    public async Task Health_returns_ok_json()
    {
        var response = await _client.GetAsync("/health");
        response.EnsureSuccessStatusCode();

        var payload = await response.Content.ReadFromJsonAsync<HealthPayload>();
        Assert.NotNull(payload);
        Assert.Equal("ok", payload!.status);
        Assert.Equal("D-1", payload.phase);
    }

    [Fact]
    public async Task Status_returns_worker_and_task_counts()
    {
        var response = await _client.GetAsync("/status");
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync();
        Assert.Contains("tasks", body);
        Assert.Contains("workers", body);
    }

    [Fact]
    public async Task Register_without_api_key_is_rejected()
    {
        var response = await _client.PostAsJsonAsync("/register", new
        {
            workerName = "unused",
            enrollmentKey = "",
            processArchitecture = "X64",
            osDescription = "TestOS",
            capability = new
            {
                tokensPerSecond = 1000d,
                cpuThreads = 4,
                calibrationDurationMs = 1000,
                benchmarkId = "Int8TernaryMatMul",
                measuredAt = DateTimeOffset.UtcNow
            }
        });

        // Missing X-Api-Key => 401 Unauthorized from the ApiKey
        // scheme's challenge handler.
        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task Work_without_api_key_is_rejected()
    {
        var response = await _client.GetAsync("/work");
        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task Work_with_wrong_api_key_is_rejected()
    {
        using var message = new HttpRequestMessage(HttpMethod.Get, "/work");
        message.Headers.Add("X-Api-Key", "not-the-real-key");
        message.Headers.Add("X-Worker-Id", "worker-1");

        using var response = await _client.SendAsync(message);
        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task Work_with_correct_api_key_returns_204_for_empty_queue()
    {
        using var message = new HttpRequestMessage(HttpMethod.Get, "/work");
        message.Headers.Add("X-Api-Key", TestApiKey);
        message.Headers.Add("X-Worker-Id", "worker-smoke");

        using var response = await _client.SendAsync(message);
        Assert.Equal(HttpStatusCode.NoContent, response.StatusCode);
    }

    [Fact]
    public async Task Admin_dashboard_without_cookie_triggers_oidc_challenge()
    {
        var response = await _client.GetAsync("/admin/dashboard");

        // The cookie + OIDC chain produces a 302 redirect to either
        // the OIDC challenge endpoint or directly to the login page.
        Assert.Equal(HttpStatusCode.Redirect, response.StatusCode);
        Assert.NotNull(response.Headers.Location);
    }

    /// <summary>
    /// Minimal deserialization target for <c>/health</c>. Having it
    /// inline keeps the test class self-contained.
    /// </summary>
    private sealed record HealthPayload(string status, DateTimeOffset time, string phase);
}
#endif
