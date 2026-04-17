#if NET10_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Track 7 end-to-end tests for <c>GET /weights/{version}</c>.
/// Validates that the coordinator serves a <see cref="WeightBlobCodec"/>-
/// encoded blob whose decoded fp32 length matches the configured
/// preset's <see cref="BitNetSharp.Core.Training.FlatParameterPack.ComputeLength"/>.
/// This is the wire contract the worker relies on in its flat-param
/// training loop.
/// </summary>
public sealed class WeightsRouteTests
    : IClassFixture<WeightsRouteTests.WeightsFactory>, IDisposable
{
    private const string TestApiKey = "test-shared-worker-key";

    // Track 7 canonical value: 5174 * 256 + 4 * (4*256*256 + 2*256*1024 + 1024*256) + 256 * 5174
    private const int ExpectedSmallPresetFlatLength = 6_843_392;

    public sealed class WeightsFactory : WebApplicationFactory<CoordinatorHostMarker>
    {
        public string WorkDirectory { get; } =
            Path.Combine(Path.GetTempPath(), $"bitnet-coord-weights-{Guid.NewGuid():N}");

        public string DatabasePath => Path.Combine(WorkDirectory, "coordinator.db");

        public WeightsFactory()
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
                    ["Coordinator:Admin:Username"] = "admin",
                    ["Coordinator:Admin:Password"] = "super-secret-test",
                    ["Coordinator:WorkerApiKey"] = TestApiKey,
                    ["Coordinator:ModelPreset"] = "small"
                };
                config.AddInMemoryCollection(testSettings);
            });
        }
    }

    private readonly WeightsFactory _factory;
    private readonly HttpClient _client;

    public WeightsRouteTests(WeightsFactory factory)
    {
        _factory = factory;
        _client = _factory.CreateClient(new WebApplicationFactoryClientOptions
        {
            AllowAutoRedirect = false
        });
    }

    public void Dispose() => _client.Dispose();

    [Fact]
    public async Task Get_weights_v1_returns_full_flat_vector_for_small_preset()
    {
        using var message = new HttpRequestMessage(HttpMethod.Get, "/weights/1");
        message.Headers.Add("X-Api-Key", TestApiKey);
        message.Headers.Add("X-Worker-Id", "worker-weights-smoke");

        using var response = await _client.SendAsync(message);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/octet-stream", response.Content.Headers.ContentType?.MediaType);

        var bytes = await response.Content.ReadAsByteArrayAsync();
        var expectedSize = WeightBlobCodec.HeaderSize + 4L * ExpectedSmallPresetFlatLength;
        Assert.Equal(expectedSize, bytes.LongLength);

        Assert.True(
            WeightBlobCodec.TryDecode(bytes, out var version, out var weights, out var error),
            $"Blob failed to decode: {error}");
        Assert.Equal(1, version);
        Assert.Equal(ExpectedSmallPresetFlatLength, weights.Length);
    }

    [Fact]
    public async Task Get_weights_without_api_key_is_rejected()
    {
        using var response = await _client.GetAsync("/weights/1");
        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task Get_weights_unknown_version_returns_404()
    {
        using var message = new HttpRequestMessage(HttpMethod.Get, "/weights/99999");
        message.Headers.Add("X-Api-Key", TestApiKey);
        message.Headers.Add("X-Worker-Id", "worker-weights-404");

        using var response = await _client.SendAsync(message);
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }
}
#endif
