using System.Reflection;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// G-series ternary dot kernel microbenchmarks. Measures Scalar oracle vs
/// the generic Vector&lt;T&gt; path (pre-G baseline) vs the AVX2 VPSIGNB
/// kernel and the VNNI VPDPBSSD variants when the host supports them.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 3, iterationCount: 10)]
public class TritDotBenchmarks
{
    [Params(64, 128, 4096, 11008)]
    public int Length;

    private sbyte[] _trits = null!;
    private sbyte[] _activations = null!;

    private static readonly MethodInfo ScalarMethod =
        typeof(TritPacking).GetMethod(
            "TernaryDotScalar",
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!;

    private static readonly MethodInfo GenericMethod =
        typeof(TritPacking).GetMethod(
            "TernaryDotSimdUnpackedGeneric",
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)!;

    private static readonly Func<ReadOnlySpan<sbyte>, ReadOnlySpan<sbyte>, int> ScalarDelegate =
        (Func<ReadOnlySpan<sbyte>, ReadOnlySpan<sbyte>, int>)
            ScalarMethod.CreateDelegate(typeof(Func<ReadOnlySpan<sbyte>, ReadOnlySpan<sbyte>, int>));

    private static readonly Func<ReadOnlySpan<sbyte>, ReadOnlySpan<sbyte>, int> GenericDelegate =
        (Func<ReadOnlySpan<sbyte>, ReadOnlySpan<sbyte>, int>)
            GenericMethod.CreateDelegate(typeof(Func<ReadOnlySpan<sbyte>, ReadOnlySpan<sbyte>, int>));

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(31337 + Length);
        _trits = new sbyte[Length];
        _activations = new sbyte[Length];
        for (var i = 0; i < Length; i++)
        {
            _trits[i] = (sbyte)(rng.Next(3) - 1);
            _activations[i] = (sbyte)(rng.Next(255) - 127);
        }
    }

    [Benchmark(Baseline = true)]
    public int Scalar() => ScalarDelegate(_trits, _activations);

    [Benchmark]
    public int Generic() => GenericDelegate(_trits, _activations);

    [Benchmark]
    public int Avx2Sign() => TritPacking.TernaryDotAvx2Sign(_trits, _activations);

    [Benchmark]
    public int AvxVnniInt8() => TritPacking.TernaryDotAvxVnniInt8(_trits, _activations);

    [Benchmark]
    public int AvxVnniInt8V512() => TritPacking.TernaryDotAvxVnniInt8V512(_trits, _activations);

    [Benchmark]
    public int Dispatcher() => TritPacking.TernaryDotSimdUnpacked(_trits, _activations);
}
