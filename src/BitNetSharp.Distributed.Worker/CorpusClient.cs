using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;
using BitNetSharp.Distributed.Contracts;
using Serilog;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Fetches tokenized corpus shards from the coordinator's
/// <c>GET /corpus/{shardId}</c> endpoint via HTTP Range requests.
/// <para>
/// Shards on the wire are a raw little-endian <c>int32</c> token
/// stream (same format as <c>data/WikiText2/wikitext-2-valid-tokens.bin</c>).
/// The coordinator serves them with <c>enableRangeProcessing: true</c>
/// so a worker can slice out the exact byte window described by
/// <see cref="WorkTaskAssignment.ShardOffset"/> /
/// <see cref="WorkTaskAssignment.ShardLength"/> without pulling the
/// entire shard.
/// </para>
/// <para>
/// A small LRU keyed by <c>(shardId, offset, length)</c> coalesces
/// consecutive tasks that hit the same byte window. Keep the cap
/// low (default 2) — each entry can be multiple MiB.
/// </para>
/// </summary>
internal sealed class CorpusClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttpClient;
    private readonly int _maxCachedShards;
    private readonly object _cacheGate = new();
    private readonly Dictionary<CacheKey, LinkedListNode<CacheEntry>> _cacheIndex = new();
    private readonly LinkedList<CacheEntry> _cacheOrder = new();

    public CorpusClient(HttpClient http, bool ownsHttpClient, int maxCachedShards = 2)
    {
        _http = http ?? throw new ArgumentNullException(nameof(http));
        _ownsHttpClient = ownsHttpClient;
        _maxCachedShards = Math.Max(1, maxCachedShards);
    }

    /// <summary>
    /// Parses a raw byte buffer into <c>int32</c> tokens in
    /// little-endian order, rounding the length down to the nearest
    /// multiple of 4 if the buffer is ragged. Ragged inputs are
    /// logged at Warning since they usually indicate a caller bug.
    /// </summary>
    public static int[] ParseTokens(ReadOnlySpan<byte> bytes)
    {
        var usable = bytes.Length - (bytes.Length % sizeof(int));
        if (usable != bytes.Length)
        {
            Log.Warning(
                "CorpusClient.ParseTokens: dropping {Trailing} trailing byte(s) — {Total} is not a multiple of 4",
                bytes.Length - usable,
                bytes.Length);
        }

        var count = usable / sizeof(int);
        var tokens = new int[count];
        for (var i = 0; i < count; i++)
        {
            tokens[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes.Slice(i * sizeof(int), sizeof(int)));
        }
        return tokens;
    }

    /// <summary>
    /// Fetches bytes <c>[byteOffset, byteOffset + byteLength)</c> of
    /// <paramref name="shardId"/> from the coordinator. Results are
    /// cached by <c>(shardId, offset, length)</c>; a repeat call with
    /// the same tuple short-circuits without an HTTP round trip.
    /// Non-success responses surface as <see cref="HttpRequestException"/>.
    /// </summary>
    public async Task<byte[]> FetchShardRangeAsync(
        string shardId,
        long byteOffset,
        long byteLength,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(shardId))
        {
            throw new ArgumentException("shardId must not be empty.", nameof(shardId));
        }
        if (byteOffset < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteOffset), byteOffset, "must be non-negative");
        }
        if (byteLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(byteLength), byteLength, "must be positive");
        }

        var key = new CacheKey(shardId, byteOffset, byteLength);
        if (TryTakeFromCache(key, out var cached))
        {
            return cached;
        }

        using var request = new HttpRequestMessage(HttpMethod.Get, $"corpus/{shardId}");
        request.Headers.Range = new RangeHeaderValue(byteOffset, byteOffset + byteLength - 1);

        using var response = await _http.SendAsync(
                request,
                HttpCompletionOption.ResponseContentRead,
                cancellationToken)
            .ConfigureAwait(false);

        response.EnsureSuccessStatusCode();

        var bytes = await response.Content
            .ReadAsByteArrayAsync(cancellationToken)
            .ConfigureAwait(false);

        StoreInCache(key, bytes);
        return bytes;
    }

    /// <summary>
    /// Convenience wrapper: fetches the task's shard window, parses
    /// to <c>int32</c> tokens, and chunks into non-overlapping
    /// sequences of length <paramref name="maxSequenceLength"/>.
    /// Trailing tokens that don't fill a sequence are dropped. If
    /// fewer than one full sequence is available the return value
    /// is empty (the worker upstream will skip the task).
    /// </summary>
    public async Task<IReadOnlyList<int[]>> FetchSequencesAsync(
        WorkTaskAssignment task,
        int maxSequenceLength,
        CancellationToken cancellationToken)
    {
        if (task is null) throw new ArgumentNullException(nameof(task));
        if (maxSequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), maxSequenceLength, "must be positive");
        }

        var bytes = await FetchShardRangeAsync(task.ShardId, task.ShardOffset, task.ShardLength, cancellationToken)
            .ConfigureAwait(false);

        var tokens = ParseTokens(bytes);
        if (tokens.Length < maxSequenceLength)
        {
            return Array.Empty<int[]>();
        }

        var sequenceCount = tokens.Length / maxSequenceLength;
        var sequences = new int[sequenceCount][];
        for (var i = 0; i < sequenceCount; i++)
        {
            var seq = new int[maxSequenceLength];
            Array.Copy(tokens, i * maxSequenceLength, seq, 0, maxSequenceLength);
            sequences[i] = seq;
        }
        return sequences;
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _http.Dispose();
        }
    }

    // ── Cache plumbing ────────────────────────────────────────────

    private bool TryTakeFromCache(CacheKey key, out byte[] bytes)
    {
        lock (_cacheGate)
        {
            if (_cacheIndex.TryGetValue(key, out var node))
            {
                // Promote to MRU position on hit.
                _cacheOrder.Remove(node);
                _cacheOrder.AddFirst(node);
                bytes = node.Value.Bytes;
                return true;
            }
        }
        bytes = Array.Empty<byte>();
        return false;
    }

    private void StoreInCache(CacheKey key, byte[] bytes)
    {
        lock (_cacheGate)
        {
            if (_cacheIndex.TryGetValue(key, out var existing))
            {
                _cacheOrder.Remove(existing);
                _cacheIndex.Remove(key);
            }

            var node = new LinkedListNode<CacheEntry>(new CacheEntry(key, bytes));
            _cacheOrder.AddFirst(node);
            _cacheIndex[key] = node;

            while (_cacheOrder.Count > _maxCachedShards)
            {
                var victim = _cacheOrder.Last!;
                _cacheOrder.RemoveLast();
                _cacheIndex.Remove(victim.Value.Key);
            }
        }
    }

    private readonly record struct CacheKey(string ShardId, long Offset, long Length);
    private readonly record struct CacheEntry(CacheKey Key, byte[] Bytes);
}
