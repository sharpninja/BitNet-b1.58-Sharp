using BenchmarkDotNet.Attributes;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// BenchmarkDotNet benchmark class that measures the throughput of the
/// integer-accumulated ternary matrix multiplication inner loop that
/// dominates BitNet training. All workers run this identical workload on
/// startup so the coordinator has a hardware-agnostic score that compares
/// apples-to-apples between a Raspberry Pi and a 64-core Threadripper.
///
/// This benchmark is intentionally self-contained — it does NOT reference
/// any BitNetSharp.Core training primitives — so it stays stable as the
/// training kernels evolve. The coordinator calibration multiplier folds in
/// the difference between raw matmul throughput and full forward+backward
/// step cost.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class WorkerCapabilityBenchmark
{
    /// <summary>
    /// Sequences per invocation. Small enough that even a very slow worker
    /// completes the calibration in under a minute.
    /// </summary>
    internal const int Batch = 4;

    /// <summary>Tokens per sequence.</summary>
    internal const int SequenceLength = 64;

    /// <summary>
    /// Hidden dimension: must match the representative inner loop width of
    /// a ~100M BitNet SLM. 512 is the target for the TruckMate scale.
    /// </summary>
    internal const int Hidden = 512;

    /// <summary>Output projection width (attention or FFN).</summary>
    internal const int OutputWidth = 512;

    /// <summary>
    /// Tokens processed per benchmark invocation, used by BenchmarkDotNet to
    /// normalize the reported metric so the median is nanoseconds-per-token.
    /// </summary>
    public const int TokensPerInvoke = Batch * SequenceLength;

    private sbyte[] _activations = null!;
    private sbyte[] _weights = null!;
    private int[] _output = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(1337);
        _activations = new sbyte[Batch * SequenceLength * Hidden];
        _weights = new sbyte[OutputWidth * Hidden];
        _output = new int[Batch * SequenceLength * OutputWidth];

        for (int i = 0; i < _activations.Length; i++)
        {
            _activations[i] = (sbyte)rng.Next(-127, 128);
        }

        for (int i = 0; i < _weights.Length; i++)
        {
            _weights[i] = (sbyte)(rng.Next(3) - 1);
        }
    }

    /// <summary>
    /// Representative BitNet training inner loop: int8 activations dot
    /// ternary {-1,0,+1} weights accumulated into int32. Mirrors the hot
    /// path that <c>BitNetSharp.Core</c>'s BitLinear layer executes for
    /// every forward pass. The method is deliberately written without
    /// intrinsics so RyuJIT's auto-vectorizer can pick whatever ISA level
    /// the host CPU exposes; this is what makes the benchmark fair across
    /// hardware that supports different SIMD widths.
    /// </summary>
    [Benchmark(OperationsPerInvoke = TokensPerInvoke)]
    public long Int8TernaryMatMul()
    {
        var activations = _activations;
        var weights = _weights;
        var output = _output;
        long checksum = 0;

        for (int token = 0; token < TokensPerInvoke; token++)
        {
            int activationRow = token * Hidden;
            int outputRow = token * OutputWidth;

            for (int outputColumn = 0; outputColumn < OutputWidth; outputColumn++)
            {
                int weightRow = outputColumn * Hidden;
                int accumulator = 0;

                for (int k = 0; k < Hidden; k++)
                {
                    accumulator += activations[activationRow + k] * weights[weightRow + k];
                }

                output[outputRow + outputColumn] = accumulator;
                checksum += accumulator;
            }
        }

        // Returned so RyuJIT cannot eliminate the entire loop as dead code.
        return checksum;
    }
}
