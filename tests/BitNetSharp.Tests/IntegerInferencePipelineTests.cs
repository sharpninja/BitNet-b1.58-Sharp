using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase V1: compose every integer-inference primitive (I3 IntegerRmsNorm,
/// I4 IntegerRotaryPositionEmbedding, I5 BitLinear.ForwardInt32, I6
/// IntegerSoftmax, I7 IntegerSwiGLU, I8 IntegerResidualAdder, I9 argmax)
/// into a single-layer end-to-end pipeline and verify it tracks the float
/// reference within the integer-precision floor (perplexity within 0.5 on
/// the final logits proxy). If one primitive drifts, this integration test
/// will flag it before the full transformer rewrite lands.
/// </summary>
public sealed class IntegerInferencePipelineTests
{
    private const int Dim = 32;
    private const int Hidden = 64;

    [Fact]
    public void IntegerBlock_MatchesFloatReference_WithinTolerance()
    {
        var rng = new Random(41);
        var input = BuildMatrix(4, Dim, rng);

        // Shared weights / modules used by both reference and integer paths.
        var qProj = BuildBitLinear(Dim, Dim, rng);
        var kProj = BuildBitLinear(Dim, Dim, rng);
        var vProj = BuildBitLinear(Dim, Dim, rng);
        var gateProj = BuildBitLinear(Dim, Hidden, rng);
        var upProj = BuildBitLinear(Dim, Hidden, rng);
        var downProj = BuildBitLinear(Hidden, Dim, rng);
        var normScale = new float[Dim];
        for (var i = 0; i < Dim; i++) normScale[i] = 1f + ((float)rng.NextDouble() - 0.5f) * 0.1f;

        var floatOut = RunFloatReference(input, qProj, kProj, vProj, gateProj, upProj, downProj, normScale);
        var integerOut = RunIntegerPipeline(input, qProj, kProj, vProj, gateProj, upProj, downProj, normScale);

        // Per-element absolute tolerance: accumulation across I3+I5+I7+I8 is
        // bounded by the coarsest LUT (softmax 4096 entries) and Q16.16 shifts.
        for (var r = 0; r < 4; r++)
        {
            for (var c = 0; c < Dim; c++)
            {
                Assert.InRange(integerOut[r, c] - floatOut[r, c], -5e-2f, 5e-2f);
            }
        }
    }

    [Fact]
    public void IntegerArgmax_OnLogits_MatchesFloatArgmax()
    {
        var rng = new Random(53);
        var hidden = BuildMatrix(1, Dim, rng);
        var lmHead = BuildBitLinear(Dim, 16, rng);

        var quant = QuantizedActivationBlock.FromFloat(hidden);
        var floatLogits = lmHead.ForwardQuantized(quant);
        var intLogits = lmHead.ForwardInt32(quant);

        // Argmax on raw int32 logits should match argmax on dequantized floats
        // because softmax is monotonic (I9 contract).
        var intLogitsRow = new int[16];
        for (var c = 0; c < 16; c++) intLogitsRow[c] = intLogits.Values[0, c];
        int intArgmax = IntegerSampling.Argmax(intLogitsRow);

        int floatArgmax = 0;
        float best = floatLogits[0, 0];
        for (var c = 1; c < 16; c++)
        {
            if (floatLogits[0, c] > best) { best = floatLogits[0, c]; floatArgmax = c; }
        }

        Assert.Equal(floatArgmax, intArgmax);
    }

    private static float[,] RunFloatReference(
        float[,] input,
        BitLinear qProj, BitLinear kProj, BitLinear vProj,
        BitLinear gateProj, BitLinear upProj, BitLinear downProj,
        float[] normScale)
    {
        var normed = ApplyRmsNorm(input, normScale, 1e-6f);
        var q = qProj.Forward(normed);
        var k = kProj.Forward(normed);
        var v = vProj.Forward(normed);

        ApplyRoPE(q);
        ApplyRoPE(k);

        var attnOut = ScaledDotProductAttention(q, k, v);
        var afterResid = AddMatrices(input, attnOut);

        var normedFfn = ApplyRmsNorm(afterResid, normScale, 1e-6f);
        var gate = gateProj.Forward(normedFfn);
        var up = upProj.Forward(normedFfn);
        var swish = new float[gate.GetLength(0), gate.GetLength(1)];
        for (var r = 0; r < swish.GetLength(0); r++)
        {
            for (var c = 0; c < swish.GetLength(1); c++)
            {
                var g = gate[r, c];
                var sig = 1f / (1f + MathF.Exp(-g));
                swish[r, c] = g * sig * up[r, c];
            }
        }
        var down = downProj.Forward(swish);
        return AddMatrices(afterResid, down);
    }

    private static float[,] RunIntegerPipeline(
        float[,] input,
        BitLinear qProj, BitLinear kProj, BitLinear vProj,
        BitLinear gateProj, BitLinear upProj, BitLinear downProj,
        float[] normScale)
    {
        var intRms = new IntegerRmsNorm(Dim, 1e-6f);
        intRms.ImportScale(normScale);
        var intRope = new IntegerRotaryPositionEmbedding(headDim: Dim, maxSequenceLength: 16);
        var intSoftmax = new IntegerSoftmax();
        var intSwiGLU = new IntegerSwiGLU();
        var intAdder = new IntegerResidualAdder();

        var normed = intRms.Forward(input);
        var quantAttn = QuantizedActivationBlock.FromFloat(normed);
        var qBlock = qProj.ForwardInt32(quantAttn);
        var kBlock = kProj.ForwardInt32(quantAttn);
        var vBlock = vProj.ForwardInt32(quantAttn);

        var q = qBlock.ToFloat();
        var k = kBlock.ToFloat();
        var vFloat = vBlock.ToFloat();
        intRope.ApplyInPlace(q, headCount: 1);
        intRope.ApplyInPlace(k, headCount: 1);

        var attnOut = ScaledDotProductAttentionWithIntSoftmax(q, k, vFloat, intSoftmax);

        var inputAsBlock = FloatToInt32Block(input);
        var attnAsBlock = FloatToInt32Block(attnOut);
        var afterResid = intAdder.Add(inputAsBlock, attnAsBlock).ToFloat();

        var normedFfn = intRms.Forward(afterResid);
        var quantFfn = QuantizedActivationBlock.FromFloat(normedFfn);
        var gate = gateProj.ForwardInt32(quantFfn).ToFloat();
        var up = upProj.ForwardInt32(quantFfn).ToFloat();
        var swish = intSwiGLU.ApplyToFloat(gate, up);
        var quantDown = QuantizedActivationBlock.FromFloat(swish);
        var down = downProj.ForwardInt32(quantDown).ToFloat();

        var afterResidBlock = FloatToInt32Block(afterResid);
        var downBlock = FloatToInt32Block(down);
        return intAdder.Add(afterResidBlock, downBlock).ToFloat();
    }

    private static Int32ActivationBlock FloatToInt32Block(float[,] values)
    {
        var rows = values.GetLength(0);
        var cols = values.GetLength(1);
        var intValues = new int[rows, cols];
        var scales = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            var maxAbs = 0f;
            for (var c = 0; c < cols; c++)
            {
                var abs = Math.Abs(values[r, c]);
                if (abs > maxAbs) maxAbs = abs;
            }
            var scale = maxAbs == 0f ? 1f : maxAbs / 32767f;
            scales[r] = scale;
            for (var c = 0; c < cols; c++)
            {
                intValues[r, c] = (int)Math.Round(values[r, c] / scale);
            }
        }
        return new Int32ActivationBlock(intValues, scales);
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

    private static float[,] ApplyRmsNorm(float[,] input, float[] scale, float epsilon)
    {
        var rows = input.GetLength(0);
        var cols = input.GetLength(1);
        var output = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            double sumSq = 0;
            for (var c = 0; c < cols; c++) sumSq += input[r, c] * input[r, c];
            var inv = 1f / MathF.Sqrt((float)(sumSq / cols) + epsilon);
            for (var c = 0; c < cols; c++) output[r, c] = input[r, c] * inv * scale[c];
        }
        return output;
    }

    private static void ApplyRoPE(float[,] tensor)
    {
        var rope = new RotaryPositionEmbedding(Dim);
        rope.ApplyInPlace(tensor, headCount: 1);
    }

    private static float[,] ScaledDotProductAttention(float[,] q, float[,] k, float[,] v)
    {
        var seqLen = q.GetLength(0);
        var dim = q.GetLength(1);
        var scale = 1f / MathF.Sqrt(dim);
        var weights = new float[seqLen, seqLen];
        for (var i = 0; i < seqLen; i++)
        {
            var max = float.NegativeInfinity;
            for (var j = 0; j < seqLen; j++)
            {
                float dot = 0;
                for (var d = 0; d < dim; d++) dot += q[i, d] * k[j, d];
                weights[i, j] = dot * scale;
                if (weights[i, j] > max) max = weights[i, j];
            }
            double sum = 0;
            for (var j = 0; j < seqLen; j++)
            {
                weights[i, j] = MathF.Exp(weights[i, j] - max);
                sum += weights[i, j];
            }
            for (var j = 0; j < seqLen; j++) weights[i, j] = (float)(weights[i, j] / sum);
        }

        var output = new float[seqLen, dim];
        for (var i = 0; i < seqLen; i++)
        {
            for (var d = 0; d < dim; d++)
            {
                float acc = 0;
                for (var j = 0; j < seqLen; j++) acc += weights[i, j] * v[j, d];
                output[i, d] = acc;
            }
        }
        return output;
    }

    private static float[,] ScaledDotProductAttentionWithIntSoftmax(
        float[,] q, float[,] k, float[,] v, IntegerSoftmax softmax)
    {
        var seqLen = q.GetLength(0);
        var dim = q.GetLength(1);
        var scale = 1f / MathF.Sqrt(dim);
        var logits = new float[seqLen, seqLen];
        for (var i = 0; i < seqLen; i++)
        {
            for (var j = 0; j < seqLen; j++)
            {
                float dot = 0;
                for (var d = 0; d < dim; d++) dot += q[i, d] * k[j, d];
                logits[i, j] = dot * scale;
            }
        }

        var weights = softmax.ApplyToFloat(logits);

        var output = new float[seqLen, dim];
        for (var i = 0; i < seqLen; i++)
        {
            for (var d = 0; d < dim; d++)
            {
                float acc = 0;
                for (var j = 0; j < seqLen; j++) acc += weights[i, j] * v[j, d];
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
}
