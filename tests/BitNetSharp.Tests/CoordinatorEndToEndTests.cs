#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Threading.Tasks;
using Duende.IdentityServer.Services;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator;
using BitNetSharp.Distributed.Coordinator.Persistence;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authentication.OpenIdConnect;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// End-to-end integration test exercising the full worker lifecycle
/// against the real coordinator Web host via
/// <see cref="WebApplicationFactory{TEntryPoint}"/>. Walks through
/// /connect/token → /register → /heartbeat → /work → /gradient and
/// asserts the SQLite task state is Done after the round trip.
///
/// <para>
/// The critical trick is wiring both JwtBearer and OpenIdConnect
/// middleware to use the TestServer's own message handler as their
/// OIDC discovery backchannel. Without that override they would try
/// to dial out to the configured Authority URL
/// (<c>http://localhost</c>) and time out because the in-process
/// test host is not listening on a real port.
/// </para>
/// </summary>
public sealed class CoordinatorEndToEndTests : IClassFixture<CoordinatorEndToEndTests.E2EFactory>, IDisposable
{
    public sealed class E2EFactory : WebApplicationFactory<CoordinatorHostMarker>
    {
        public string DatabasePath { get; } =
            Path.Combine(Path.GetTempPath(), $"bitnet-coord-e2e-{Guid.NewGuid():N}.db");

        public string WeightsDirectory => Path.Combine(Path.GetDirectoryName(DatabasePath) ?? ".", "weights");

        public const string ClientId = "worker-e2e";
        public const string ClientSecret = "e2e-secret-change-me";
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
                    ["Coordinator:AccessTokenLifetimeSeconds"] = "3600",
                    ["Coordinator:Admin:Username"] = AdminUsername,
                    ["Coordinator:Admin:Password"] = AdminPassword,
                    ["Coordinator:WorkerClients:0:ClientId"] = ClientId,
                    ["Coordinator:WorkerClients:0:ClientSecret"] = ClientSecret,
                    ["Coordinator:WorkerClients:0:DisplayName"] = "E2E Worker"
                };
                config.AddInMemoryCollection(testSettings);
            });

            builder.ConfigureTestServices(services =>
            {
                // Redirect the JwtBearer and OpenIdConnect backchannel
                // HTTP clients so they loop back into the TestServer
                // instead of trying to reach the configured Authority
                // URL over a real network. Server is not available
                // yet when these callbacks are registered — we grab it
                // lazily on first option resolution.
                services.AddSingleton<IPostConfigureOptions<JwtBearerOptions>>(sp =>
                    new BackchannelPostConfigure<JwtBearerOptions>(
                        "Bearer",
                        () => Server.CreateHandler(),
                        (o, h) => o.BackchannelHttpHandler = h));

                services.AddSingleton<IPostConfigureOptions<OpenIdConnectOptions>>(sp =>
                    new BackchannelPostConfigure<OpenIdConnectOptions>(
                        OpenIdConnectDefaults.AuthenticationScheme,
                        () => Server.CreateHandler(),
                        (o, h) => o.BackchannelHttpHandler = h));
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

        /// <summary>
        /// Minimal <see cref="IPostConfigureOptions{TOptions}"/> that
        /// overrides an auth scheme's <c>BackchannelHttpHandler</c>
        /// from a lazy factory. Used for both JwtBearer and OIDC.
        /// </summary>
        private sealed class BackchannelPostConfigure<TOptions> : IPostConfigureOptions<TOptions>
            where TOptions : class
        {
            private readonly string _targetScheme;
            private readonly Func<HttpMessageHandler> _handlerFactory;
            private readonly Action<TOptions, HttpMessageHandler> _apply;

            public BackchannelPostConfigure(
                string targetScheme,
                Func<HttpMessageHandler> handlerFactory,
                Action<TOptions, HttpMessageHandler> apply)
            {
                _targetScheme = targetScheme;
                _handlerFactory = handlerFactory;
                _apply = apply;
            }

            public void PostConfigure(string? name, TOptions options)
            {
                if (name == _targetScheme)
                {
                    _apply(options, _handlerFactory());
                }
            }
        }
    }

    private readonly E2EFactory _factory;
    private readonly HttpClient _client;

    public CoordinatorEndToEndTests(E2EFactory factory)
    {
        _factory = factory;

        // Force the in-process TestServer to come up so we can grab
        // its message handler and hand it to the JwtBearer +
        // OpenIdConnect middleware as their OIDC-discovery backchannel.
        // Without this override the middleware would try to resolve
        // the Authority URL over a real network and fail — the whole
        // integration test depends on the loopback path.
        _ = _factory.Server;
        var handler = _factory.Server.CreateHandler();

        var jwtOptions = _factory.Services
            .GetRequiredService<IOptionsMonitor<JwtBearerOptions>>()
            .Get("Bearer");

        // Bypass OIDC discovery entirely by pulling Duende's current
        // signing keys out of its IKeyMaterialService and stuffing
        // them into JwtBearer's TokenValidationParameters. The
        // BackchannelHttpHandler + ConfigurationManager loopback path
        // turned out to be brittle under TestServer because the
        // ConfigurationManager captures a separate HttpDocumentRetriever
        // on the first fetch, so the cleanest test fix is to short-
        // circuit discovery altogether. In production the coordinator
        // still uses the proper discovery flow.
        var keyMaterial = _factory.Services.GetRequiredService<IKeyMaterialService>();
        var validationKeys = keyMaterial.GetValidationKeysAsync().GetAwaiter().GetResult();
        jwtOptions.TokenValidationParameters.IssuerSigningKeys =
            validationKeys.Select(k => k.Key).ToList();
        jwtOptions.TokenValidationParameters.ValidateIssuer = false;
        jwtOptions.TokenValidationParameters.ValidateAudience = false;
        jwtOptions.ConfigurationManager = null;
        jwtOptions.RequireHttpsMetadata = false;

        var oidcOptions = _factory.Services
            .GetRequiredService<IOptionsMonitor<OpenIdConnectOptions>>()
            .Get(OpenIdConnectDefaults.AuthenticationScheme);
        oidcOptions.BackchannelHttpHandler = handler;
        oidcOptions.Backchannel = new HttpClient(handler, disposeHandler: false);
        oidcOptions.ConfigurationManager = null;

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

    private async Task<string> GetAccessTokenAsync()
    {
        var tokenRequest = new HttpRequestMessage(HttpMethod.Post, "/connect/token")
        {
            Content = new FormUrlEncodedContent(new[]
            {
                new KeyValuePair<string, string>("grant_type",    "client_credentials"),
                new KeyValuePair<string, string>("client_id",     E2EFactory.ClientId),
                new KeyValuePair<string, string>("client_secret", E2EFactory.ClientSecret),
                new KeyValuePair<string, string>("scope",         "bitnet-worker")
            })
        };

        using var response = await _client.SendAsync(tokenRequest);
        var rawBody = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode)
        {
            throw new InvalidOperationException(
                $"Token request returned {(int)response.StatusCode}: {rawBody}");
        }

        var payload = System.Text.Json.JsonSerializer.Deserialize<TokenResponse>(rawBody);
        Assert.NotNull(payload);
        Assert.False(string.IsNullOrWhiteSpace(payload!.AccessToken));
        return payload.AccessToken!;
    }

    private HttpRequestMessage Authed(HttpMethod method, string path, string token, HttpContent? body = null)
    {
        var req = new HttpRequestMessage(method, path);
        req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
        if (body is not null)
        {
            req.Content = body;
        }
        return req;
    }

    [Fact]
    public async Task Full_worker_lifecycle_registers_claims_and_completes_a_task()
    {
        // 0. Sanity-check that the test's configured WorkerClients
        //    were picked up by the WorkerClientRegistry singleton.
        var registry = _factory.Services.GetRequiredService<BitNetSharp.Distributed.Coordinator.Identity.WorkerClientRegistry>();
        Assert.NotNull(registry.Find(E2EFactory.ClientId));

        // 1. Obtain JWT
        var token = await GetAccessTokenAsync();

        // 2. POST /register — worker comes online.
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
            Authed(HttpMethod.Post, "/register", token, JsonContent.Create(registrationRequest))))
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
            Assert.Equal(E2EFactory.ClientId, body!.WorkerId);
            Assert.Equal(150_016L, body.RecommendedTokensPerTask);
        }

        // 3. POST /heartbeat — server accepts the keep-alive.
        var heartbeat = new HeartbeatRequest(
            WorkerId: E2EFactory.ClientId,
            Status: "idle",
            CurrentTaskId: null,
            TokensSeenSinceLastHeartbeat: 0);
        using (var heartbeatResponse = await _client.SendAsync(
            Authed(HttpMethod.Post, "/heartbeat", token, JsonContent.Create(heartbeat))))
        {
            heartbeatResponse.EnsureSuccessStatusCode();
        }

        // 4. Seed one pending task directly into the queue store so
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

        // 5. GET /work — expect the task back.
        WorkTaskAssignment? assignment;
        using (var workResponse = await _client.SendAsync(
            Authed(HttpMethod.Get, "/work", token)))
        {
            Assert.Equal(HttpStatusCode.OK, workResponse.StatusCode);
            assignment = await workResponse.Content.ReadFromJsonAsync<WorkTaskAssignment>();
            Assert.NotNull(assignment);
            Assert.Equal("task-e2e-001", assignment!.TaskId);
        }

        // 6. POST /gradient — report completion.
        var submission = new GradientSubmission(
            TaskId: assignment.TaskId,
            WorkerId: E2EFactory.ClientId,
            BaseWeightVersion: assignment.WeightVersion,
            TokensSeen: assignment.TokensPerTask,
            LossAfter: 0.5,
            GradientFormat: "stub-noop",
            GradientPayload: Array.Empty<byte>(),
            WallClockMs: 500);
        using (var gradientResponse = await _client.SendAsync(
            Authed(HttpMethod.Post, "/gradient", token, JsonContent.Create(submission))))
        {
            gradientResponse.EnsureSuccessStatusCode();
        }

        // 7. Verify final state: task is Done.
        Assert.Equal(1, queueStore.CountByState(WorkTaskState.Done));
        Assert.Equal(0, queueStore.CountByState(WorkTaskState.Assigned));
        Assert.Equal(0, queueStore.CountByState(WorkTaskState.Pending));
    }

    /// <summary>
    /// Minimal shape for Duende's /connect/token JSON response.
    /// </summary>
    private sealed record TokenResponse
    {
        [System.Text.Json.Serialization.JsonPropertyName("access_token")]
        public string? AccessToken { get; init; }

        [System.Text.Json.Serialization.JsonPropertyName("expires_in")]
        public int ExpiresIn { get; init; }

        [System.Text.Json.Serialization.JsonPropertyName("token_type")]
        public string? TokenType { get; init; }
    }
}
#endif
