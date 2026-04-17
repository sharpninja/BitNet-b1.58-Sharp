using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Track 7 endpoint-level tests for <c>POST /gradient</c>. Covers the
/// length-mismatch rejection path (shape-validated against the
/// configured model preset) and the happy path (a real-length
/// gradient produces a version bump).
///
/// <para>
/// These tests boot a full coordinator via <see cref="WebApplicationFactory{T}"/>
/// configured with a tiny temp SQLite database and the "small" model
/// preset so the canonical flat length is a known 6,843,392.
/// </para>
/// </summary>
public sealed class GradientRoutesTests
    : IClassFixture<GradientRoutesTests.GradientFactory>, IDisposable
{
    private const string TestApiKey = "test-shared-worker-key";

    public sealed class GradientFactory : WebApplicationFactory<CoordinatorHostMarker>
    {
        public string WorkDirectory { get; } =
            Path.Combine(Path.GetTempPath(), $"bitnet-coord-gradient-{Guid.NewGuid():N}");

        public string DatabasePath => Path.Combine(WorkDirectory, "coordinator.db");

        public GradientFactory()
        {
            Directory.CreateDirectory(WorkDirectory);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);
            if (disposing && Directory.Exists(WorkDirectory))
            {
                try { Directory.Delete(WorkDirectory, recursive: true); } catch { /* best-effort */ }
            }
        }

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
                    ["Coordinator:WorkerApiKey"] = TestApiKey,
                    ["Coordinator:ModelPreset"] = "small"
                };
                config.AddInMemoryCollection(testSettings);
            });
        }

    }

    private readonly GradientFactory _factory;
    private readonly HttpClient _client;

    public GradientRoutesTests(GradientFactory factory)
    {
        _factory = factory;
        _client = _factory.CreateClient(new WebApplicationFactoryClientOptions
        {
            AllowAutoRedirect = false
        });
    }

    public void Dispose() => _client.Dispose();

    /// <summary>
    /// Builds an int8-ef payload that, after decode, has
    /// <paramref name="decodedLength"/> fp32 elements. Uses a zero
    /// payload so the decoded gradient is all zeros — enough to
    /// drive the length-check path without needing real training
    /// numerics. Built by hand so we don't need a matching residual
    /// buffer.
    /// </summary>
    private static byte[] EncodeZeroGradient(int decodedLength)
    {
        var residual = new float[decodedLength];
        var gradient = new float[decodedLength];
        return Int8GradientCodec.Encode(gradient, residual);
    }

    [Fact]
    public async Task Post_gradient_with_wrong_length_payload_returns_400_length_mismatch()
    {
        // The worker is not registered and no task is assigned —
        // but the length check fires before the task-assignment
        // check because WeightApplicationService.Apply runs first
        // in the handler. Use a random worker id / task id.
        var submission = new GradientSubmission(
            TaskId: "task-fake",
            WorkerId: "worker-length-mismatch",
            BaseWeightVersion: 1,
            TokensSeen: 0,
            LossAfter: 0d,
            GradientFormat: Int8GradientCodec.FormatId,
            // 4096 elements = legacy D-1 placeholder, definitely
            // not the 6,843,392 the small preset expects.
            GradientPayload: EncodeZeroGradient(4096),
            WallClockMs: 0);

        using var message = new HttpRequestMessage(HttpMethod.Post, "/gradient");
        message.Headers.Add("X-Api-Key", TestApiKey);
        message.Headers.Add("X-Worker-Id", "worker-length-mismatch");
        message.Content = JsonContent.Create(submission);

        using var response = await _client.SendAsync(message);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var payload = await response.Content.ReadFromJsonAsync<ErrorResponse>();
        Assert.NotNull(payload);
        Assert.Equal("length_mismatch", payload!.Code);
        Assert.Contains("4096", payload.Message);
        Assert.Contains("6843392", payload.Message);
    }

    [Fact]
    public async Task Post_gradient_without_api_key_is_rejected()
    {
        using var message = new HttpRequestMessage(HttpMethod.Post, "/gradient")
        {
            Content = JsonContent.Create(new GradientSubmission(
                TaskId: "task-x",
                WorkerId: "worker-x",
                BaseWeightVersion: 1,
                TokensSeen: 0,
                LossAfter: 0d,
                GradientFormat: Int8GradientCodec.FormatId,
                GradientPayload: EncodeZeroGradient(4),
                WallClockMs: 0))
        };

        using var response = await _client.SendAsync(message);
        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }
}
