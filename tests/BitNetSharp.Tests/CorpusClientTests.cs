using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Worker;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests for <see cref="CorpusClient"/>. Covers the
/// pure byte-to-int32 parse path, the Range-fetch HTTP contract
/// (header correctness + body pass-through), and the LRU shard
/// cache that coalesces two identical fetches into a single
/// upstream HTTP round-trip.
/// </summary>
public sealed class CorpusClientTests
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

    private sealed class RecordingHandler : HttpMessageHandler
    {
        public List<HttpRequestMessage> Requests { get; } = new();
        public Func<HttpRequestMessage, HttpResponseMessage> Responder { get; set; } =
            _ => new HttpResponseMessage(HttpStatusCode.InternalServerError);

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            return Task.FromResult(Responder(request));
        }
    }

    private static CorpusClient CreateClient(RecordingHandler handler, WorkerConfig config, int maxCachedShards = 2)
    {
        var http = new HttpClient(handler)
        {
            BaseAddress = config.CoordinatorUrl
        };
        http.DefaultRequestHeaders.Add(CoordinatorClient.ApiKeyHeader, config.ApiKey);
        http.DefaultRequestHeaders.Add(CoordinatorClient.WorkerIdHeader, config.WorkerId);
        return new CorpusClient(http, ownsHttpClient: true, maxCachedShards: maxCachedShards);
    }

    private static byte[] EncodeInt32LittleEndian(int[] tokens)
    {
        var bytes = new byte[tokens.Length * sizeof(int)];
        for (var i = 0; i < tokens.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(
                bytes.AsSpan(i * sizeof(int), sizeof(int)),
                tokens[i]);
        }
        return bytes;
    }

    [Fact]
    public void ParseTokens_roundtrips_eight_little_endian_int32_values()
    {
        var expected = new[] { 1, 29871, 13, 353, -1, 0, 2147483647, -2147483648 };
        var bytes = EncodeInt32LittleEndian(expected);

        var actual = CorpusClient.ParseTokens(bytes);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ParseTokens_rounds_down_to_int32_multiple_with_warning_logged()
    {
        // 9 bytes → 2 complete int32s + 1 trailing byte dropped.
        var full = EncodeInt32LittleEndian(new[] { 42, 99 });
        var ragged = new byte[full.Length + 1];
        Array.Copy(full, ragged, full.Length);
        ragged[^1] = 0xFF;

        var actual = CorpusClient.ParseTokens(ragged);

        Assert.Equal(new[] { 42, 99 }, actual);
    }

    [Fact]
    public async Task FetchShardRangeAsync_sets_Range_header_and_returns_raw_bytes()
    {
        var tokens = new[] { 10, 20, 30, 40, 50, 60 };
        var allBytes = EncodeInt32LittleEndian(tokens);
        // slice: bytes 8..15 inclusive → tokens[2..3] = {30, 40}
        var sliceOffset = 8L;
        var sliceLength = 8L;
        var expectedSlice = allBytes.Skip((int)sliceOffset).Take((int)sliceLength).ToArray();

        var handler = new RecordingHandler
        {
            Responder = _ =>
            {
                var response = new HttpResponseMessage(HttpStatusCode.PartialContent)
                {
                    Content = new ByteArrayContent(expectedSlice)
                };
                response.Content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
                return response;
            }
        };

        using var client = CreateClient(handler, SampleConfig());
        var result = await client.FetchShardRangeAsync("shard-42", sliceOffset, sliceLength, CancellationToken.None);

        Assert.Equal(expectedSlice, result);
        Assert.Single(handler.Requests);
        var req = handler.Requests[0];
        Assert.Equal(HttpMethod.Get, req.Method);
        Assert.EndsWith("/corpus/shard-42", req.RequestUri!.AbsolutePath);
        Assert.NotNull(req.Headers.Range);
        var range = req.Headers.Range!;
        Assert.Equal("bytes", range.Unit);
        var r = range.Ranges.Single();
        Assert.Equal(sliceOffset, r.From);
        Assert.Equal(sliceOffset + sliceLength - 1, r.To);
    }

    [Fact]
    public async Task FetchShardRangeAsync_caches_by_range_and_short_circuits_second_call()
    {
        var tokens = Enumerable.Range(1, 16).ToArray();
        var allBytes = EncodeInt32LittleEndian(tokens);

        var handler = new RecordingHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.PartialContent)
            {
                Content = new ByteArrayContent(allBytes)
            }
        };

        using var client = CreateClient(handler, SampleConfig(), maxCachedShards: 2);

        var first = await client.FetchShardRangeAsync("shard-a", 0, allBytes.Length, CancellationToken.None);
        var second = await client.FetchShardRangeAsync("shard-a", 0, allBytes.Length, CancellationToken.None);

        Assert.Equal(first, second);
        Assert.Single(handler.Requests); // cache hit — only one upstream HTTP call
    }

    [Fact]
    public async Task FetchShardRangeAsync_evicts_oldest_entry_when_cache_full()
    {
        var bytesA = EncodeInt32LittleEndian(new[] { 1, 2 });
        var bytesB = EncodeInt32LittleEndian(new[] { 3, 4 });
        var bytesC = EncodeInt32LittleEndian(new[] { 5, 6 });

        var idx = 0;
        var payloads = new[] { bytesA, bytesB, bytesC, bytesA };
        var handler = new RecordingHandler
        {
            Responder = _ =>
            {
                var payload = payloads[idx++];
                return new HttpResponseMessage(HttpStatusCode.PartialContent)
                {
                    Content = new ByteArrayContent(payload)
                };
            }
        };

        using var client = CreateClient(handler, SampleConfig(), maxCachedShards: 2);

        _ = await client.FetchShardRangeAsync("shard-a", 0, bytesA.Length, CancellationToken.None);
        _ = await client.FetchShardRangeAsync("shard-b", 0, bytesB.Length, CancellationToken.None);
        _ = await client.FetchShardRangeAsync("shard-c", 0, bytesC.Length, CancellationToken.None); // evicts shard-a
        _ = await client.FetchShardRangeAsync("shard-a", 0, bytesA.Length, CancellationToken.None); // must refetch

        Assert.Equal(4, handler.Requests.Count);
    }

    [Fact]
    public async Task FetchSequencesAsync_chunks_bytes_into_max_sequence_length_chunks()
    {
        // modelConfig small preset has MaxSequenceLength 256; use a
        // small fake config to keep the test tight: 16 tokens and
        // MaxSequenceLength=4 → 4 sequences of length 4.
        var tokens = Enumerable.Range(1, 16).ToArray();
        var allBytes = EncodeInt32LittleEndian(tokens);

        var handler = new RecordingHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.PartialContent)
            {
                Content = new ByteArrayContent(allBytes)
            }
        };
        using var client = CreateClient(handler, SampleConfig());

        var task = new WorkTaskAssignment(
            TaskId: "task-1",
            WeightVersion: 1,
            WeightUrl: "/weights/1",
            ShardId: "shard-fetch",
            ShardOffset: 0,
            ShardLength: allBytes.Length,
            TokensPerTask: 16,
            KLocalSteps: 1,
            HyperparametersJson: "{}",
            DeadlineUtc: DateTimeOffset.UtcNow.AddMinutes(10));

        var sequences = await client.FetchSequencesAsync(task, maxSequenceLength: 4, CancellationToken.None);

        Assert.Equal(4, sequences.Count);
        Assert.All(sequences, seq => Assert.Equal(4, seq.Length));
        Assert.Equal(new[] { 1, 2, 3, 4 }, sequences[0]);
        Assert.Equal(new[] { 13, 14, 15, 16 }, sequences[3]);
    }

    [Fact]
    public async Task FetchSequencesAsync_returns_empty_when_token_count_below_sequence_length()
    {
        var allBytes = EncodeInt32LittleEndian(new[] { 1, 2 }); // only 2 tokens
        var handler = new RecordingHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.PartialContent)
            {
                Content = new ByteArrayContent(allBytes)
            }
        };
        using var client = CreateClient(handler, SampleConfig());

        var task = new WorkTaskAssignment(
            TaskId: "task-small",
            WeightVersion: 1,
            WeightUrl: "/weights/1",
            ShardId: "shard-small",
            ShardOffset: 0,
            ShardLength: allBytes.Length,
            TokensPerTask: 2,
            KLocalSteps: 1,
            HyperparametersJson: "{}",
            DeadlineUtc: DateTimeOffset.UtcNow.AddMinutes(10));

        var sequences = await client.FetchSequencesAsync(task, maxSequenceLength: 4, CancellationToken.None);

        Assert.Empty(sequences);
    }

    [Fact]
    public async Task FetchSequencesAsync_rounds_down_ragged_byte_length_to_int32_multiple()
    {
        // ShardLength not a multiple of 4 → parser drops the tail
        // byte and sequences come back as if length had been floored.
        var tokens = Enumerable.Range(1, 8).ToArray();
        var full = EncodeInt32LittleEndian(tokens);
        var ragged = new byte[full.Length + 3]; // odd trailing bytes
        Array.Copy(full, ragged, full.Length);
        var handler = new RecordingHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.PartialContent)
            {
                Content = new ByteArrayContent(ragged)
            }
        };
        using var client = CreateClient(handler, SampleConfig());

        var task = new WorkTaskAssignment(
            TaskId: "task-ragged",
            WeightVersion: 1,
            WeightUrl: "/weights/1",
            ShardId: "shard-ragged",
            ShardOffset: 0,
            ShardLength: ragged.Length,
            TokensPerTask: 8,
            KLocalSteps: 1,
            HyperparametersJson: "{}",
            DeadlineUtc: DateTimeOffset.UtcNow.AddMinutes(10));

        var sequences = await client.FetchSequencesAsync(task, maxSequenceLength: 4, CancellationToken.None);

        Assert.Equal(2, sequences.Count);
        Assert.Equal(new[] { 1, 2, 3, 4 }, sequences[0]);
        Assert.Equal(new[] { 5, 6, 7, 8 }, sequences[1]);
    }

    [Fact]
    public async Task FetchShardRangeAsync_throws_when_shard_is_missing()
    {
        var handler = new RecordingHandler
        {
            Responder = _ => new HttpResponseMessage(HttpStatusCode.NotFound)
            {
                Content = new StringContent("{\"error\":\"unknown_shard\"}", Encoding.UTF8, "application/json")
            }
        };
        using var client = CreateClient(handler, SampleConfig());

        await Assert.ThrowsAsync<HttpRequestException>(async () =>
            await client.FetchShardRangeAsync("missing", 0, 16, CancellationToken.None));
    }
}
