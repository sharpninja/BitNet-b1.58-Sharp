using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Full-sequence attention vs. cached-decode attention. Cached variant prefills
/// the first (SeqLen-1) tokens once into the KV cache and then measures a single
/// decode step (1-row query against SeqLen cached K/V rows). That is the hot
/// loop for autoregressive generation.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 3, iterationCount: 10)]
public class AttentionBenchmarks
{
    [Params(32, 128, 512)]
    public int SeqLen;

    private BitNetConfig _config = null!;
    private GroupedQueryAttention _gqa = null!;
    private float[,] _input = null!;

    private GroupedQueryAttention _gqaCached = null!;
    private LayerKvCache _cache = null!;
    private float[,] _newRow = null!;

    [GlobalSetup]
    public void Setup()
    {
        _config = TestBitNetFactory.RealisticConfig();
        _gqa = TestBitNetFactory.CreateGqa(_config);
        _input = TestBitNetFactory.RandomActivations(SeqLen, _config.Dimension, seed: 31);

        _gqaCached = TestBitNetFactory.CreateGqa(_config);
        var kvDim = _config.KvHeadCount * _config.HeadDimension;
        _cache = new LayerKvCache(SeqLen + 1, kvDim);
        var prefill = new float[SeqLen, _config.Dimension];
        for (var r = 0; r < SeqLen; r++)
        {
            for (var c = 0; c < _config.Dimension; c++)
            {
                prefill[r, c] = _input[r, c];
            }
        }
        _gqaCached.Forward(prefill, _cache, positionOffset: 0);

        _newRow = TestBitNetFactory.RandomActivations(1, _config.Dimension, seed: 131);
    }

    [Benchmark(Baseline = true)]
    public float[,] Forward_FullSequence() => _gqa.Forward(_input);

    [Benchmark]
    public float[,] Forward_CachedDecode() => _gqaCached.Forward(_newRow, _cache, positionOffset: SeqLen);

    [Benchmark]
    public float[,] Forward_FlashDecode() => _gqaCached.ForwardFlashDecode(_newRow, _cache, positionOffset: SeqLen);
}
