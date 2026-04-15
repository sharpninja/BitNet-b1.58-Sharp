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
/// unauthenticated traffic. Full JWT + OIDC issuance tests will land
/// once the BackchannelHttpHandler wiring for the in-process Duende
/// discovery document is in place — see the integration blocker
/// noted in the session log.
/// </summary>
public sealed class CoordinatorEndpointTests : IClassFixture<CoordinatorEndpointTests.CoordinatorFactory>, IDisposable
{
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
                    ["Coordinator:WorkerClients:0:ClientId"] = "test-worker",
                    ["Coordinator:WorkerClients:0:ClientSecret"] = "test-secret",
                    ["Coordinator:WorkerClients:0:DisplayName"] = "Test Worker"
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
        // The single configured worker client must show up in the
        // /status dashboard so the operator can confirm their env
        // vars took effect.
        Assert.Contains("\"configured\":1", body);
    }

    [Fact]
    public async Task Register_without_JWT_returns_challenge()
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

        // The WorkerPolicy requires a Bearer JWT; without one the
        // authorization middleware either returns 401 directly or
        // 302 redirecting to the cookie login path depending on
        // which scheme's challenge wins. Either signals "auth needed".
        Assert.True(
            response.StatusCode == HttpStatusCode.Unauthorized
            || response.StatusCode == HttpStatusCode.Redirect
            || response.StatusCode == HttpStatusCode.Forbidden,
            $"Expected auth challenge; got {(int)response.StatusCode} {response.StatusCode}");
    }

    [Fact]
    public async Task Work_without_JWT_returns_challenge()
    {
        var response = await _client.GetAsync("/work");
        Assert.True(
            response.StatusCode == HttpStatusCode.Unauthorized
            || response.StatusCode == HttpStatusCode.Redirect
            || response.StatusCode == HttpStatusCode.Forbidden,
            $"Expected auth challenge; got {(int)response.StatusCode} {response.StatusCode}");
    }

    [Fact]
    public async Task AdminApiKeys_without_cookie_triggers_oidc_challenge()
    {
        var response = await _client.GetAsync("/admin/api-keys");

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
