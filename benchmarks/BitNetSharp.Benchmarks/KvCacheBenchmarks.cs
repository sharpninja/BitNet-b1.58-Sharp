using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Benchmarks;

/// <summary>
/// Section B (KV cache quantization) - KV0 scaffolding.
/// Isolates the dot-against-KV-cache portion of attention so the bandwidth
/// win from int8 K/V can be read without end-to-end attention noise. The
/// fp32 variants run against existing <see cref="LayerKvCache"/>; the int8
/// variants currently NotImplemented and will go green as KV1-KV4 land.
/// Bonsai shape: dim=4096, kvHeads=8, headDim=128, kvDim=1024.
/// </summary>
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net10_0, warmupCount: 3, iterationCount: 10)]
public class KvCacheBenchmarks
{
    [Params(32, 128, 512, 2048)]
    public int SeqLen;

    private BitNetConfig _config = null!;
    private LayerKvCache _cacheFp32 = null!;
    private QuantizedKvLayerCache _cacheInt8 = null!;
    private float[] _query = null!;

    [GlobalSetup]
    public void Setup()
    {
        _config = TestBitNetFactory.RealisticConfig();
        var kvDim = _config.KvHeadCount * _config.HeadDimension;
        _cacheFp32 = new LayerKvCache(SeqLen, kvDim);
        _cacheInt8 = new QuantizedKvLayerCache(SeqLen, kvDim);

        var rng = new Random(31);
        var kRow = new float[kvDim];
        var vRow = new float[kvDim];
        for (var r = 0; r < SeqLen; r++)
        {
            for (var c = 0; c < kvDim; c++)
            {
                kRow[c] = (float)(rng.NextDouble() * 2.0 - 1.0);
                vRow[c] = (float)(rng.NextDouble() * 2.0 - 1.0);
                _cacheFp32.K[r, c] = kRow[c];
                _cacheFp32.V[r, c] = vRow[c];
            }
            _cacheInt8.WriteRow(r, kRow, vRow);
        }

        _query = new float[_config.HeadCount * _config.HeadDimension];
        for (var i = 0; i < _query.Length; i++)
        {
            _query[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }
    }

    /// <summary>
    /// Sums QK and AV over all cached rows for head 0. Baseline: fp32 K/V.
    /// </summary>
    [Benchmark(Baseline = true)]
    public float DotScan_Fp32()
    {
        var headDim = _config.HeadDimension;
        var qSlice = _query.AsSpan(0, headDim);
        var sum = 0f;

        var kFlat = AttentionMath.AsFlatSpan(_cacheFp32.K);
        for (var row = 0; row < SeqLen; row++)
        {
            var kSlice = kFlat.Slice(row * _cacheFp32.KvDimension, headDim);
            sum += AttentionMath.Dot(qSlice, kSlice, headDim);
        }
        return sum;
    }

    /// <summary>
    /// Same scan but against an int8 K cache with per-row absmax scale.
    /// </summary>
    [Benchmark]
    public float DotScan_Int8()
    {
        var headDim = _config.HeadDimension;
        var qSlice = _query.AsSpan(0, headDim);
        var sum = 0f;

        ref var kFirst = ref System.Runtime.CompilerServices.Unsafe.As<byte, sbyte>(
            ref System.Runtime.InteropServices.MemoryMarshal.GetArrayDataReference(_cacheInt8.K));
        var kFlat = System.Runtime.InteropServices.MemoryMarshal.CreateSpan(ref kFirst, _cacheInt8.K.Length);

        for (var row = 0; row < SeqLen; row++)
        {
            var kSlice = kFlat.Slice(row * _cacheInt8.KvDimension, headDim);
            sum += AttentionMath.DotInt8(qSlice, kSlice, _cacheInt8.KScale[row], headDim);
        }
        return sum;
    }
}
