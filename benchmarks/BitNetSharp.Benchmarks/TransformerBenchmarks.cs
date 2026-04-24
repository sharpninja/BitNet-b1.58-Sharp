using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Transformer forward cost. Baseline re-processes every position; cached
/// decode re-uses KV for prior positions and only processes the new token.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 2, iterationCount: 5)]
public class TransformerBenchmarks
{
    [Params(16, 64, 128)]
    public int SeqLen;

    private BitNetTransformer _transformer = null!;
    private int[] _tokens = null!;

    private BitNetTransformer _transformerCached = null!;
    private TransformerCache _cache = null!;
    private int[] _decodeToken = null!;

    [GlobalSetup]
    public void Setup()
    {
        var config = TestBitNetFactory.SmallConfig();
        _transformer = new BitNetTransformer(config, NullLogger<BitNetTransformer>.Instance, seed: 42);
        var rng = new Random(17);
        _tokens = new int[SeqLen];
        for (var i = 0; i < SeqLen; i++)
        {
            _tokens[i] = rng.Next(0, config.VocabSize);
        }

        _transformerCached = new BitNetTransformer(config, NullLogger<BitNetTransformer>.Instance, seed: 42);
        _cache = _transformerCached.CreateCache(SeqLen + 1);
        _transformerCached.Forward(_tokens, _cache);
        _decodeToken = new[] { rng.Next(0, config.VocabSize) };
    }

    [Benchmark(Baseline = true)]
    public float[,] Forward_Full() => _transformer.Forward(_tokens);

    [Benchmark]
    public float[,] Forward_CachedDecode()
    {
        // Reset cache to prefilled state: we only want to measure a single decode
        // step (position SeqLen) against the cached [0, SeqLen) past.
        _cache.RollbackTo(SeqLen);
        return _transformerCached.Forward(_decodeToken, _cache);
    }
}
