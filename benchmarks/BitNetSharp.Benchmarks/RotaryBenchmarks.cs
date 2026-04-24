using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// RoPE ApplyInPlace cost baseline. Phase 1 adds a variant with positionOffset
/// to show the per-decode saving from not rebuilding sin/cos tables for the
/// entire sequence.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 3, iterationCount: 10)]
public class RotaryBenchmarks
{
    [Params(1, 32, 128)]
    public int SeqLen;

    private const int HeadDim = 128;
    private const int HeadCount = 32;

    private RotaryPositionEmbedding _rope = null!;
    private float[,] _tensor = null!;

    [GlobalSetup]
    public void Setup()
    {
        _rope = new RotaryPositionEmbedding(HeadDim, theta: 1_000_000d);
        _tensor = TestBitNetFactory.RandomActivations(SeqLen, HeadCount * HeadDim, seed: 1337);
    }

    [IterationSetup]
    public void IterationSetup()
    {
        // ApplyInPlace mutates; regenerate each iteration for determinism.
        for (var r = 0; r < SeqLen; r++)
        {
            for (var c = 0; c < HeadCount * HeadDim; c++)
            {
                _tensor[r, c] = ((r * 31 + c) % 1024) / 512f - 1f;
            }
        }
    }

    [Benchmark(Baseline = true)]
    public void ApplyInPlace_FullSequence() => _rope.ApplyInPlace(_tensor, HeadCount);
}
