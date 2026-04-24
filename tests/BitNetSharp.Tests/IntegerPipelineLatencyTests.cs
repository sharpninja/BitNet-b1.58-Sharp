using System.Diagnostics;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase V2: integer-inference latency budget. At a realistic BitNet layer
/// shape (dim=512, hidden=1536, seq=32) the composed integer pipeline for
/// one attention+FFN block must complete well under the 2 s/token budget so
/// that a 36-layer Bonsai-scale decode can land inside it end-to-end. This
/// is the micro-benchmark that protects the per-primitive speed targets; the
/// full-transformer gate lives in the Ollama integration run.
/// </summary>
public sealed class IntegerPipelineLatencyTests
{
    private const int Dim = 512;
    private const int Hidden = 1536;
    private const int SeqLen = 32;

    [Fact]
    public void OneBlock_IntegerPipeline_UnderSixtyMilliseconds()
    {
        var rng = new Random(59);
        var input = BuildMatrix(SeqLen, Dim, rng);

        var qProj = BuildBitLinear(Dim, Dim, rng);
        var kProj = BuildBitLinear(Dim, Dim, rng);
        var vProj = BuildBitLinear(Dim, Dim, rng);
        var gateProj = BuildBitLinear(Dim, Hidden, rng);
        var upProj = BuildBitLinear(Dim, Hidden, rng);
        var downProj = BuildBitLinear(Hidden, Dim, rng);

        var normScale = new float[Dim];
        for (var i = 0; i < Dim; i++) normScale[i] = 1f;
        var intRms = new IntegerRmsNorm(Dim, 1e-6f);
        intRms.ImportScale(normScale);
        var intRope = new IntegerRotaryPositionEmbedding(headDim: Dim, maxSequenceLength: 128);
        var intSoftmax = new IntegerSoftmax();
        var intSwiGLU = new IntegerSwiGLU();

        // Warm up JIT / LUT access patterns.
        RunOneBlock(input, qProj, kProj, vProj, gateProj, upProj, downProj,
            intRms, intRope, intSoftmax, intSwiGLU);

        var sw = Stopwatch.StartNew();
        RunOneBlock(input, qProj, kProj, vProj, gateProj, upProj, downProj,
            intRms, intRope, intSoftmax, intSwiGLU);
        sw.Stop();

        // 36-layer Bonsai target is <2 s/token at seq=100. Per-block budget
        // at seq=32 is generous: 60 ms leaves ~2.2 s headroom for a 36-layer
        // transformer even before cache reuse kicks in.
        Assert.True(sw.ElapsedMilliseconds < 60,
            $"Integer one-block forward took {sw.ElapsedMilliseconds} ms (budget 60 ms).");
    }

    private static void RunOneBlock(
        float[,] input,
        BitLinear qProj, BitLinear kProj, BitLinear vProj,
        BitLinear gateProj, BitLinear upProj, BitLinear downProj,
        IntegerRmsNorm intRms,
        IntegerRotaryPositionEmbedding intRope,
        IntegerSoftmax intSoftmax,
        IntegerSwiGLU intSwiGLU)
    {
        var normed = intRms.Forward(input);
        var quantAttn = QuantizedActivationBlock.FromFloat(normed);
        var qBlock = qProj.ForwardInt32(quantAttn);
        var kBlock = kProj.ForwardInt32(quantAttn);
        var vBlock = vProj.ForwardInt32(quantAttn);

        var q = qBlock.ToFloat();
        var k = kBlock.ToFloat();
        var v = vBlock.ToFloat();
        intRope.ApplyInPlace(q, headCount: 1);
        intRope.ApplyInPlace(k, headCount: 1);

        var logits = BuildAttentionLogits(q, k);
        var weights = intSoftmax.ApplyToFloat(logits);
        var attnOut = WeightedSum(weights, v);

        var residual = AddMatrices(input, attnOut);
        var normedFfn = intRms.Forward(residual);
        var quantFfn = QuantizedActivationBlock.FromFloat(normedFfn);
        var gate = gateProj.ForwardInt32(quantFfn).ToFloat();
        var up = upProj.ForwardInt32(quantFfn).ToFloat();
        var swish = intSwiGLU.ApplyToFloat(gate, up);
        var quantDown = QuantizedActivationBlock.FromFloat(swish);
        var down = downProj.ForwardInt32(quantDown).ToFloat();
        _ = AddMatrices(residual, down);
    }

    private static float[,] BuildAttentionLogits(float[,] q, float[,] k)
    {
        var seq = q.GetLength(0);
        var dim = q.GetLength(1);
        var scale = 1f / MathF.Sqrt(dim);
        var logits = new float[seq, seq];
        for (var i = 0; i < seq; i++)
        {
            for (var j = 0; j < seq; j++)
            {
                float dot = 0;
                for (var d = 0; d < dim; d++) dot += q[i, d] * k[j, d];
                logits[i, j] = dot * scale;
            }
        }
        return logits;
    }

    private static float[,] WeightedSum(float[,] weights, float[,] v)
    {
        var seq = weights.GetLength(0);
        var dim = v.GetLength(1);
        var output = new float[seq, dim];
        for (var i = 0; i < seq; i++)
        {
            for (var d = 0; d < dim; d++)
            {
                float acc = 0;
                for (var j = 0; j < seq; j++) acc += weights[i, j] * v[j, d];
                output[i, d] = acc;
            }
        }
        return output;
    }

    private static float[,] AddMatrices(float[,] a, float[,] b)
    {
        var rows = a.GetLength(0);
        var cols = a.GetLength(1);
        var output = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++) output[r, c] = a[r, c] + b[r, c];
        }
        return output;
    }

    private static float[,] BuildMatrix(int rows, int cols, Random rng)
    {
        var m = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                m[r, c] = ((float)rng.NextDouble() - 0.5f) * 2f;
            }
        }
        return m;
    }

    private static BitLinear BuildBitLinear(int inDim, int outDim, Random rng)
    {
        var layer = new BitLinear(new BitLinearConfig(inDim, outDim));
        var weights = new float[outDim, inDim];
        for (var r = 0; r < outDim; r++)
        {
            for (var c = 0; c < inDim; c++)
            {
                weights[r, c] = ((float)rng.NextDouble() - 0.5f) * 2f;
            }
        }
        layer.QuantizeFromFullPrecision(weights);
        return layer;
    }
}
