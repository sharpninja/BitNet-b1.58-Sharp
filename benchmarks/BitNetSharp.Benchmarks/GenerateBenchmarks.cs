using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// End-to-end prompt + N-token decode. The non-cached baseline re-processes
/// the growing context every step; the cached variant prefills once and emits
/// one new token per forward. SmallConfig keeps the benchmark tractable.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 1, iterationCount: 3)]
public class GenerateBenchmarks
{
    [Params(8, 16)]
    public int PromptLen;

    [Params(4, 8)]
    public int NewTokens;

    private BitNetTransformer _transformer = null!;
    private int[] _prompt = null!;
    private int[] _step = null!;
    private int _capacity;

    [GlobalSetup]
    public void Setup()
    {
        var config = TestBitNetFactory.SmallConfig();
        _transformer = new BitNetTransformer(config, NullLogger<BitNetTransformer>.Instance, seed: 42);
        var rng = new Random(19);
        _prompt = new int[PromptLen];
        for (var i = 0; i < PromptLen; i++)
        {
            _prompt[i] = rng.Next(0, config.VocabSize);
        }
        _step = new[] { rng.Next(0, config.VocabSize) };
        _capacity = PromptLen + NewTokens + 1;
    }

    [Benchmark(Baseline = true)]
    public float[,] Generate_FullRecompute()
    {
        var context = new List<int>(_prompt);
        float[,] logits = _transformer.Forward(context);
        for (var i = 0; i < NewTokens; i++)
        {
            context.Add(_step[0]);
            logits = _transformer.Forward(context);
        }

        return logits;
    }

    [Benchmark]
    public float[,] Generate_KvCache()
    {
        var cache = _transformer.CreateCache(_capacity);
        var logits = _transformer.Forward(_prompt, cache);
        for (var i = 0; i < NewTokens; i++)
        {
            logits = _transformer.Forward(_step, cache);
        }

        return logits;
    }
}
