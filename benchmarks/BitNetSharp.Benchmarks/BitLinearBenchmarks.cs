using System.Reflection;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Baseline BitLinear.Forward cost across (rows, in_dim, out_dim). Rows=1
/// models the decode-time hot path; rows=32/128 the prefill path.
/// Out_dim 14336 matches the FFN hidden dimension used by Bonsai.
/// G-series adds <see cref="ForwardQuantizedForcedGeneric"/> so the BDN
/// report shows the BitLinear-level delta of the accelerated dispatch
/// against the pre-G generic path directly.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 3, iterationCount: 10)]
public class BitLinearBenchmarks
{
    [Params(1, 32, 128)]
    public int Rows;

    [Params(512, 4096)]
    public int InDim;

    [Params(512, 4096, 14336)]
    public int OutDim;

    private BitLinear _layer = null!;
    private float[,] _input = null!;
    private QuantizedActivationBlock _quant = null!;
    private static readonly FieldInfo ForceGenericField =
        typeof(TritPacking).Assembly.GetType("BitNetSharp.Core.Quantization.TritDotDispatch")!
            .GetField("ForceGeneric", BindingFlags.NonPublic | BindingFlags.Static)!;

    [GlobalSetup]
    public void Setup()
    {
        _layer = ParameterInitializer.CreateBitLinear(new BitLinearConfig(InDim, OutDim), new Random(42));
        _input = TestBitNetFactory.RandomActivations(Rows, InDim, seed: 17);
        _quant = QuantizedActivationBlock.FromFloat(_input);
    }

    [Benchmark(Baseline = true)]
    public float[,] Forward() => _layer.Forward(_input);

    [Benchmark]
    public float[,] ForwardQuantized() => _layer.ForwardQuantized(_quant);

    [Benchmark]
    public float[,] ForwardQuantizedForcedGeneric()
    {
        var orig = (bool)ForceGenericField.GetValue(null)!;
        try
        {
            ForceGenericField.SetValue(null, true);
            return _layer.ForwardQuantized(_quant);
        }
        finally
        {
            ForceGenericField.SetValue(null, orig);
        }
    }
}
