using System.Numerics;
using System.Runtime.CompilerServices;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Layers;

public sealed class BitLinear : Module
{
    private const int ActivationQuantizationMaxMagnitude = 127;
    private const float WeightQuantizationEpsilon = 1e-6f;
    private static readonly bool UseSimd = Vector.IsHardwareAccelerated && Vector<sbyte>.Count >= 16;

    private readonly sbyte[] _ternaryWeights;

    public BitLinear(BitLinearConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        Config = config;
        _ternaryWeights = new sbyte[config.OutputDimension * config.InputDimension];
    }

    public BitLinearConfig Config { get; }

    public float Gamma { get; private set; }

    public bool HasBias => false;

    public int ActivationQuantizationBound => ActivationQuantizationMaxMagnitude;

    public int ActivationQuantizationBitWidth => 8;

    public long EstimateResidentParameterBytes() =>
        ((long)_ternaryWeights.Length * sizeof(sbyte)) + sizeof(float);

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var inputDim = Config.InputDimension;

        if (input.GetLength(1) != inputDim)
        {
            throw new ArgumentException($"Expected input dimension {inputDim}, but received {input.GetLength(1)}.", nameof(input));
        }

        var (quantizedInput, rowScales) = QuantizeActivations(input);
        var rows = input.GetLength(0);
        var output = new float[rows, Config.OutputDimension];

        for (var row = 0; row < rows; row++)
        {
            var dequantScale = Gamma * rowScales[row];
            var activationOffset = row * inputDim;

            for (var outputColumn = 0; outputColumn < Config.OutputDimension; outputColumn++)
            {
                var weightSpan = _ternaryWeights.AsSpan(outputColumn * inputDim, inputDim);
                var activationSpan = quantizedInput.AsSpan(activationOffset, inputDim);

                var isum = UseSimd
                    ? TernaryDotSimd(weightSpan, activationSpan)
                    : TernaryDotScalar(weightSpan, activationSpan);

                output[row, outputColumn] = isum * dequantScale;
            }
        }

        return output;
    }

    public void QuantizeFromFullPrecision(float[,] fullPrecisionWeights)
    {
        ArgumentNullException.ThrowIfNull(fullPrecisionWeights);

        if (fullPrecisionWeights.GetLength(0) != Config.OutputDimension || fullPrecisionWeights.GetLength(1) != Config.InputDimension)
        {
            throw new ArgumentException(
                $"Expected weights with shape [{Config.OutputDimension}, {Config.InputDimension}], but received [{fullPrecisionWeights.GetLength(0)}, {fullPrecisionWeights.GetLength(1)}].",
                nameof(fullPrecisionWeights));
        }

        Gamma = ComputeAbsMean(fullPrecisionWeights);

        if (Gamma <= 0f)
        {
            Array.Clear(_ternaryWeights, 0, _ternaryWeights.Length);
            return;
        }

        var inputDim = Config.InputDimension;
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            var offset = row * inputDim;
            for (var column = 0; column < inputDim; column++)
            {
                var normalized = fullPrecisionWeights[row, column] / Gamma;
                normalized += WeightQuantizationEpsilon;
                var quantized = Math.Clamp((int)MathF.Round(normalized, MidpointRounding.AwayFromZero), -1, 1);
                _ternaryWeights[offset + column] = (sbyte)quantized;
            }
        }
    }

    public float[,] ToFullPrecision()
    {
        var result = new float[Config.OutputDimension, Config.InputDimension];
        var inputDim = Config.InputDimension;

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            var offset = row * inputDim;
            for (var column = 0; column < inputDim; column++)
            {
                result[row, column] = _ternaryWeights[offset + column] * Gamma;
            }
        }

        return result;
    }

    public TernaryWeightStats GetTernaryStats()
    {
        var negativeCount = 0;
        var zeroCount = 0;
        var positiveCount = 0;

        foreach (var value in _ternaryWeights)
        {
            switch (value)
            {
                case < 0:
                    negativeCount++;
                    break;
                case > 0:
                    positiveCount++;
                    break;
                default:
                    zeroCount++;
                    break;
            }
        }

        return new TernaryWeightStats(negativeCount, zeroCount, positiveCount);
    }

    private static float ComputeAbsMean(float[,] weights)
    {
        if (weights.Length == 0)
        {
            return 0f;
        }

        var sum = 0f;
        foreach (var weight in weights)
        {
            sum += MathF.Abs(weight);
        }

        return sum / weights.Length;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int TernaryDotScalar(ReadOnlySpan<sbyte> weights, ReadOnlySpan<sbyte> activations)
    {
        var sum = 0;
        for (var i = 0; i < weights.Length; i++)
        {
            var w = weights[i];
            if (w > 0) sum += activations[i];
            else if (w < 0) sum -= activations[i];
        }

        return sum;
    }

    private static int TernaryDotSimd(ReadOnlySpan<sbyte> weights, ReadOnlySpan<sbyte> activations)
    {
        var vectorSize = Vector<sbyte>.Count;
        var positiveOne = new Vector<sbyte>(1);
        var negativeOne = new Vector<sbyte>(-1);
        var accumPos = Vector<short>.Zero;
        var accumNeg = Vector<short>.Zero;
        var i = 0;

        for (; i + vectorSize <= weights.Length; i += vectorSize)
        {
            var wVec = new Vector<sbyte>(weights.Slice(i));
            var aVec = new Vector<sbyte>(activations.Slice(i));

            var posMask = Vector.Equals(wVec, positiveOne);
            var negMask = Vector.Equals(wVec, negativeOne);

            var posVals = Vector.ConditionalSelect(posMask, aVec, Vector<sbyte>.Zero);
            var negVals = Vector.ConditionalSelect(negMask, aVec, Vector<sbyte>.Zero);

            // Widen sbyte to short for safe accumulation
            Vector.Widen(posVals, out var posLo, out var posHi);
            Vector.Widen(negVals, out var negLo, out var negHi);

            accumPos += posLo + posHi;
            accumNeg += negLo + negHi;
        }

        // Reduce short vectors to scalar via widening to int
        var result = 0;
        for (var j = 0; j < Vector<short>.Count; j++)
        {
            result += accumPos[j] - accumNeg[j];
        }

        // Scalar tail
        for (; i < weights.Length; i++)
        {
            var w = weights[i];
            if (w > 0) result += activations[i];
            else if (w < 0) result -= activations[i];
        }

        return result;
    }

    private static (sbyte[] quantized, float[] rowScales) QuantizeActivations(float[,] input)
    {
        var rows = input.GetLength(0);
        var cols = input.GetLength(1);
        var quantized = new sbyte[rows * cols];
        var rowScales = new float[rows];

        for (var row = 0; row < rows; row++)
        {
            var maxAbs = 0f;
            for (var column = 0; column < cols; column++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(input[row, column]));
            }

            if (maxAbs <= 0f)
            {
                rowScales[row] = 1f;
                continue;
            }

            var scale = maxAbs / ActivationQuantizationMaxMagnitude;
            rowScales[row] = scale;

            var offset = row * cols;
            for (var column = 0; column < cols; column++)
            {
                var q = (int)MathF.Round(input[row, column] / scale, MidpointRounding.AwayFromZero);
                quantized[offset + column] = (sbyte)Math.Clamp(q, -ActivationQuantizationMaxMagnitude, ActivationQuantizationMaxMagnitude);
            }
        }

        return (quantized, rowScales);
    }
}
