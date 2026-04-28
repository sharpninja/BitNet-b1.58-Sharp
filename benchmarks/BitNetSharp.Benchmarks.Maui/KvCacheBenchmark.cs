using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;

namespace BitNetSharp.Benchmarks.Maui;

/// <summary>
/// Stopwatch-based KvCacheBenchmarks port for Android. BenchmarkDotNet does
/// not run on Android (no JIT process spawning, no Process API), so this
/// custom runner does warmup + iteration loop + simple stats. Sized to the
/// truckmate-small preset (dim=256, heads=8, headDim=32, kvHeadCount=8 ->
/// kvDim=256) which is the realistic phone-deployment target. Bonsai-shape
/// numbers were retired as a benchmark axis: ~7-8 sec/token rules them out
/// for interactive use on mid-tier phones, so optimizing the cache path
/// for 1024-wide KV rows wastes engineering effort.
/// </summary>
public static class KvCacheBenchmark
{
    private const int HeadDim = 32;
    private const int KvDim = 256; // truckmate-small: kvHeadCount(8) * headDim(32)
    private const int WarmupIterations = 3;
    private const int MeasureIterations = 10;

    // Smaller seq_lens too: the small preset's MaxSequenceLength=128
    // makes a 2048-row scan irrelevant. 1024 kept as a stress probe.
    private static readonly int[] SeqLens = [16, 64, 128, 512, 1024];

    public sealed record Row(int SeqLen, double Fp32MeanNs, double Int8MeanNs, double Ratio);

    public static IReadOnlyList<Row> Run(Action<string>? log = null)
    {
        var uiLog = log;
        log = line =>
        {
            uiLog?.Invoke(line);
            // Mirror to System.Console which Android maps to logcat tag DOTNET.
            // Prefix with marker so adb logcat | grep BENCH_KV finds the run.
            Console.WriteLine($"BENCH_KV: {line}");
        };
        log($"Host caps: AdvSimd={AdvSimd.IsSupported}, AdvSimd.Arm64={AdvSimd.Arm64.IsSupported}, Avx2={Avx2.IsSupported}");
        log($"Vector<float>.IsHardwareAccelerated={System.Numerics.Vector.IsHardwareAccelerated}, Vector<float>.Count={System.Numerics.Vector<float>.Count}, Vector<sbyte>.Count={System.Numerics.Vector<sbyte>.Count}");
        log($"Runtime: {RuntimeInformation.FrameworkDescription}");
        log($"OS: {RuntimeInformation.OSDescription}");
        log($"Arch: {RuntimeInformation.OSArchitecture} / {RuntimeInformation.ProcessArchitecture}");
        log($"TruckMate-small shape: HeadDim={HeadDim}, KvDim={KvDim}; per-row absmax int8 + per-row fp32 scale.");
        log($"Warmup={WarmupIterations} Measure={MeasureIterations}");
        log("");

        var rows = new List<Row>();
        foreach (var seqLen in SeqLens)
        {
            var (fp32Cache, int8Cache, query) = BuildShape(seqLen);
            var fp32Mean = TimeFp32Scan(fp32Cache, query);
            var int8Mean = TimeInt8Scan(int8Cache, query);
            var ratio = int8Mean / fp32Mean;
            log($"SeqLen={seqLen,5}: Fp32={fp32Mean,9:F1} ns  Int8={int8Mean,9:F1} ns  ratio={ratio:F3}");
            rows.Add(new Row(seqLen, fp32Mean, int8Mean, ratio));
        }
        log("");
        log("Done. ratio < 1.0 means int8 faster than fp32 on this host.");
        return rows;
    }

    private static (LayerKvCache Fp32, QuantizedKvLayerCache Int8, float[] Query) BuildShape(int seqLen)
    {
        var fp32 = new LayerKvCache(seqLen, KvDim);
        var int8 = new QuantizedKvLayerCache(seqLen, KvDim);
        var rng = new Random(31);
        var kRow = new float[KvDim];
        var vRow = new float[KvDim];
        for (var r = 0; r < seqLen; r++)
        {
            for (var c = 0; c < KvDim; c++)
            {
                kRow[c] = (float)(rng.NextDouble() * 2.0 - 1.0);
                vRow[c] = (float)(rng.NextDouble() * 2.0 - 1.0);
                fp32.K[r, c] = kRow[c];
                fp32.V[r, c] = vRow[c];
            }
            int8.WriteRow(r, kRow, vRow);
        }
        var query = new float[HeadDim];
        for (var i = 0; i < query.Length; i++)
        {
            query[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
        return (fp32, int8, query);
    }

    private static double TimeFp32Scan(LayerKvCache cache, float[] query)
    {
        // Warmup
        for (var w = 0; w < WarmupIterations; w++) _ = Fp32ScanOnce(cache, query);

        var samples = new long[MeasureIterations];
        for (var i = 0; i < MeasureIterations; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = Fp32ScanOnce(cache, query);
            sw.Stop();
            samples[i] = sw.ElapsedTicks;
        }
        return MeanNanoseconds(samples);
    }

    private static double TimeInt8Scan(QuantizedKvLayerCache cache, float[] query)
    {
        for (var w = 0; w < WarmupIterations; w++) _ = Int8ScanOnce(cache, query);

        var samples = new long[MeasureIterations];
        for (var i = 0; i < MeasureIterations; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = Int8ScanOnce(cache, query);
            sw.Stop();
            samples[i] = sw.ElapsedTicks;
        }
        return MeanNanoseconds(samples);
    }

    private static float Fp32ScanOnce(LayerKvCache cache, float[] query)
    {
        var qSlice = ((ReadOnlySpan<float>)query).Slice(0, HeadDim);
        var sum = 0f;
        var kFlat = AttentionMath.AsFlatSpan(cache.K);
        var rows = cache.Capacity;
        for (var row = 0; row < rows; row++)
        {
            var kSlice = kFlat.Slice(row * cache.KvDimension, HeadDim);
            sum += AttentionMath.Dot(qSlice, kSlice, HeadDim);
        }
        return sum;
    }

    private static float Int8ScanOnce(QuantizedKvLayerCache cache, float[] query)
    {
        var qSlice = ((ReadOnlySpan<float>)query).Slice(0, HeadDim);
        ref var kFirst = ref System.Runtime.CompilerServices.Unsafe.As<byte, sbyte>(
            ref MemoryMarshal.GetArrayDataReference(cache.K));
        var kFlat = MemoryMarshal.CreateSpan(ref kFirst, cache.K.Length);
        var sum = 0f;
        var rows = cache.Capacity;
        for (var row = 0; row < rows; row++)
        {
            var kSlice = kFlat.Slice(row * cache.KvDimension, HeadDim);
            sum += AttentionMath.DotInt8(qSlice, kSlice, cache.KScale[row], HeadDim);
        }
        return sum;
    }

    private static double MeanNanoseconds(long[] elapsedTicks)
    {
        var totalNs = 0d;
        var ticksPerNs = (double)Stopwatch.Frequency / 1_000_000_000d;
        for (var i = 0; i < elapsedTicks.Length; i++)
        {
            totalNs += elapsedTicks[i] / ticksPerNs;
        }
        return totalNs / elapsedTicks.Length;
    }
}
