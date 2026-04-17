using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Coordinator.Configuration;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests for <see cref="SonnetAsrCorpusGenerator"/>. The generator
/// cannot be byte-deterministic — output depends on Claude Sonnet
/// sampling — so these tests validate schema/shape instead: every
/// shard line parses as <c>[USER] ... [INTENT] {...}</c>, the intent
/// JSON is well-formed, manifest counts match file counts, and the
/// cost guard + retry + missing-key paths behave as specified.
/// All tests use a stub <see cref="HttpMessageHandler"/> so no real
/// Anthropic API call is ever made.
/// </summary>
public sealed class SonnetAsrCorpusGeneratorTests : IDisposable
{
    private static readonly Regex LinePattern =
        new(@"^\[USER\] .+ \[INTENT\] \{.+\}$", RegexOptions.Compiled);

    private readonly string _root;

    public SonnetAsrCorpusGeneratorTests()
    {
        _root = Path.Combine(Path.GetTempPath(), "sonnetasr-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best-effort */ }
    }

    private static SonnetAsrCorpusGenerator NewGenerator(
        HttpMessageHandler handler,
        CoordinatorOptions? options = null)
    {
        options ??= new CoordinatorOptions
        {
            AnthropicApiKey = "sk-test-key",
            AnthropicModel = "claude-sonnet-4-6",
            AsrMaxConcurrency = 1,
            AsrCostCapUsd = 100m,
            AsrShardPrefix = "asr-v1-",
        };
        var http = new HttpClient(handler) { BaseAddress = new Uri("https://api.anthropic.com/") };
        return new SonnetAsrCorpusGenerator(
            http,
            new StaticOptionsMonitor<CoordinatorOptions>(options),
            NullLogger<SonnetAsrCorpusGenerator>.Instance);
    }

    [Fact]
    public async Task Generate_emits_well_formed_user_intent_lines()
    {
        var handler = new StubHandler(CannedOkResponse(batchLines: 10));
        var gen = NewGenerator(handler);

        var manifest = await gen.GenerateAsync(_root, count: 10, examplesPerShard: 10, seed: 1, batchSize: 10);

        var shardPath = manifest.Shards.Single().Path;
        var lines = File.ReadAllLines(shardPath);
        Assert.Equal(10, lines.Length);
        foreach (var line in lines)
        {
            Assert.Matches(LinePattern, line);
            var intentStart = line.IndexOf("[INTENT] ", StringComparison.Ordinal) + "[INTENT] ".Length;
            var intentJson = line.Substring(intentStart);
            using var doc = JsonDocument.Parse(intentJson);
            Assert.True(doc.RootElement.TryGetProperty("intent", out _));
            Assert.True(doc.RootElement.TryGetProperty("slots", out _));
        }
    }

    [Fact]
    public async Task Generate_writes_manifest_with_matching_counts()
    {
        var handler = new StubHandler(CannedOkResponse(batchLines: 5));
        var gen = NewGenerator(handler);

        var manifest = await gen.GenerateAsync(_root, count: 15, examplesPerShard: 5, seed: 2, batchSize: 5);

        Assert.Equal(15, manifest.TotalExamples);
        Assert.Equal(15, manifest.Shards.Sum(s => s.ExampleCount));
        foreach (var shard in manifest.Shards)
        {
            var actual = File.ReadAllLines(shard.Path).Length;
            Assert.Equal(shard.ExampleCount, actual);
        }
        Assert.True(File.Exists(Path.Combine(_root, "manifest.asr-v1.json")));
    }

    [Fact]
    public async Task Generate_honors_examples_per_shard()
    {
        var handler = new StubHandler(CannedOkResponse(batchLines: 10));
        var gen = NewGenerator(handler);

        var manifest = await gen.GenerateAsync(_root, count: 30, examplesPerShard: 10, seed: 3, batchSize: 10);

        Assert.Equal(3, manifest.Shards.Count);
        Assert.All(manifest.Shards, s => Assert.Equal(10, s.ExampleCount));
    }

    [Fact]
    public async Task Generate_uses_asr_v1_shard_prefix()
    {
        var handler = new StubHandler(CannedOkResponse(batchLines: 5));
        var gen = NewGenerator(handler);

        var manifest = await gen.GenerateAsync(_root, count: 5, examplesPerShard: 5, seed: 4, batchSize: 5);

        Assert.All(manifest.Shards, s => Assert.StartsWith("asr-v1-", s.ShardId));
        Assert.All(manifest.Shards, s => Assert.StartsWith("asr-v1-", Path.GetFileName(s.Path)));
    }

    [Fact]
    public async Task Generate_stops_at_cost_cap_and_returns_partial_manifest()
    {
        // Canned response reports very large token usage so cumulative
        // cost exceeds the 0.0001 USD cap after the first batch.
        var handler = new StubHandler(CannedOkResponse(batchLines: 5, inputTokens: 500_000, outputTokens: 500_000));
        var options = new CoordinatorOptions
        {
            AnthropicApiKey = "sk-test-key",
            AnthropicModel = "claude-sonnet-4-6",
            AsrMaxConcurrency = 1,
            AsrCostCapUsd = 0.0001m,
            AsrShardPrefix = "asr-v1-",
        };
        var gen = NewGenerator(handler, options);

        var manifest = await gen.GenerateAsync(_root, count: 50, examplesPerShard: 5, seed: 5, batchSize: 5);

        Assert.True(manifest.TotalExamples < 50, "should have stopped early");
        Assert.True(manifest.TotalExamples > 0, "should have produced some examples before halting");
    }

    [Fact]
    public async Task Generate_retries_on_429_then_succeeds()
    {
        var handler = new StubHandler(CannedOkResponse(batchLines: 5))
        {
            FailFirstN = 1,
            FailStatus = (HttpStatusCode)429,
        };
        var gen = NewGenerator(handler);

        var manifest = await gen.GenerateAsync(_root, count: 5, examplesPerShard: 5, seed: 6, batchSize: 5);

        Assert.Equal(5, manifest.TotalExamples);
        Assert.Equal(2, handler.CallCount);
    }

    [Fact]
    public async Task Generate_fails_fast_on_missing_api_key()
    {
        var handler = new StubHandler(CannedOkResponse(batchLines: 5));
        var options = new CoordinatorOptions
        {
            AnthropicApiKey = "",
            AnthropicModel = "claude-sonnet-4-6",
            AsrMaxConcurrency = 1,
            AsrCostCapUsd = 100m,
            AsrShardPrefix = "asr-v1-",
        };
        var gen = NewGenerator(handler, options);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            gen.GenerateAsync(_root, count: 5, examplesPerShard: 5, seed: 7, batchSize: 5));
        Assert.Equal(0, handler.CallCount);
    }

    private static string CannedOkResponse(int batchLines, long inputTokens = 100, long outputTokens = 100)
    {
        var sb = new StringBuilder();
        for (var i = 0; i < batchLines; i++)
        {
            sb.Append("[USER] take me to dallas ");
            sb.Append(i);
            sb.Append(" [INTENT] {\"intent\":\"navigate\",\"slots\":{\"destination\":\"Dallas\"}}");
            if (i < batchLines - 1) sb.Append('\n');
        }
        var textField = JsonEncodedText.Encode(sb.ToString()).ToString();
        return "{\"content\":[{\"type\":\"text\",\"text\":\"" + textField +
               "\"}],\"usage\":{\"input_tokens\":" + inputTokens +
               ",\"output_tokens\":" + outputTokens + "}}";
    }

    private sealed class StubHandler : HttpMessageHandler
    {
        private readonly string _body;

        public StubHandler(string body) { _body = body; }

        public int FailFirstN { get; set; }
        public HttpStatusCode FailStatus { get; set; } = (HttpStatusCode)429;
        public int CallCount { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            CallCount++;
            if (FailFirstN > 0)
            {
                FailFirstN--;
                return Task.FromResult(new HttpResponseMessage(FailStatus)
                {
                    Content = new StringContent("rate limited"),
                });
            }
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(_body, Encoding.UTF8, "application/json"),
            });
        }
    }
}
