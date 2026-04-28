using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Deterministic factories for small BitNet components used by the benchmark
/// suites. Avoids loading the 782 M Bonsai weights; inference-shape fidelity
/// matters more than parameter count for per-operation micro-benchmarks.
/// </summary>
public static class TestBitNetFactory
{
    public static BitNetConfig SmallConfig(int seed = 42) => new(
        vocabSize: 1024,
        dimension: 512,
        hiddenDimension: 2048,
        layerCount: 4,
        headCount: 8,
        maxSequenceLength: 1024,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 2,
        ropeTheta: 10_000f);

    /// <summary>
    /// Matches Bonsai's attention / FFN shapes (dim 4096, heads 32, kv 8,
    /// hidden 12288) but only four layers so the benchmark runs finish in
    /// seconds rather than minutes.
    /// </summary>
    public static BitNetConfig RealisticConfig(int seed = 42) => new(
        vocabSize: 4096,
        dimension: 4096,
        hiddenDimension: 12288,
        layerCount: 4,
        headCount: 32,
        maxSequenceLength: 2048,
        rmsNormEpsilon: 1e-6f,
        kvHeadCount: 8,
        ropeTheta: 1_000_000f);

    /// <summary>
    /// truckmate-small preset shape: dim=256, hidden=1024, layers=4, heads=8,
    /// seq=128, kvHeadCount=8 -> kvDim=256. Matches the actual phone-deployment
    /// target so x86 BDN numbers are 1:1 comparable to the
    /// <c>BitNetSharp.Benchmarks.Maui</c> Stopwatch numbers from the Motorola
    /// Edge 2024 logcat.
    /// </summary>
    public static BitNetConfig TruckMateSmallConfig() => new(
        vocabSize: 5174,
        dimension: 256,
        hiddenDimension: 1024,
        layerCount: 4,
        headCount: 8,
        maxSequenceLength: 128,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 8,
        ropeTheta: 10_000f);

    public static GroupedQueryAttention CreateGqa(BitNetConfig config, int seed = 42)
        => new(config, new Random(seed));

    public static MultiHeadAttention CreateMha(BitNetConfig config, int seed = 42)
        => new(config, new Random(seed));

    public static BitNetLayer CreateLayer(BitNetConfig config, int seed = 42)
        => new(config, new Random(seed));

    public static float[,] RandomActivations(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var buffer = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                buffer[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        return buffer;
    }
}
