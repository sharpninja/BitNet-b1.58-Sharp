using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Layers;

public sealed class BitLinear : Module
{
    private const int ActivationQuantizationMaxMagnitude = 127;
    private const float WeightQuantizationEpsilon = 1e-6f;
    private static readonly bool UseSimd = Vector.IsHardwareAccelerated && Vector<sbyte>.Count >= 16;

    private readonly int _totalWeights;
    private readonly int _packedStride; // packed bytes per output row
    private byte[] _packedWeights;

    // Row permutation for cache-aware token-row layout (null = identity)
    private int[]? _rowPermutation;

    // Training state (null until InitializeMasterWeights is called)
    private float[]? _masterWeights;
    private float[]? _masterGradients;
    private float[,]? _cachedInput;

    public BitLinear(BitLinearConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        Config = config;
        _totalWeights = config.OutputDimension * config.InputDimension;
        _packedStride = (config.InputDimension + 4) / 5;
        _packedWeights = new byte[config.OutputDimension * _packedStride];
    }

    public BitLinearConfig Config { get; }

    public float Gamma { get; private set; }

    public bool HasBias => false;

    public int ActivationQuantizationBound => ActivationQuantizationMaxMagnitude;

    public int ActivationQuantizationBitWidth => 8;

    public long EstimateResidentParameterBytes() =>
        (long)_packedWeights.Length + sizeof(float);

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var inputDim = Config.InputDimension;

        if (input.GetLength(1) != inputDim)
        {
            throw new ArgumentException($"Expected input dimension {inputDim}, but received {input.GetLength(1)}.", nameof(input));
        }

        if (_masterWeights is not null)
        {
            _cachedInput = (float[,])input.Clone();
        }

        var (quantizedInput, rowScales) = QuantizeActivations(input);
        var rows = input.GetLength(0);
        var output = new float[rows, Config.OutputDimension];

        var unpackBuffer = ArrayPool<sbyte>.Shared.Rent(inputDim);
        try
        {
            for (var row = 0; row < rows; row++)
            {
                var dequantScale = Gamma * rowScales[row];
                var activationOffset = row * inputDim;

                for (var outputColumn = 0; outputColumn < Config.OutputDimension; outputColumn++)
                {
                    var physicalRow = _rowPermutation is not null ? _rowPermutation[outputColumn] : outputColumn;
                    TritPacking.UnpackRowInto(_packedWeights, physicalRow * _packedStride, _packedStride, unpackBuffer, inputDim);

                    var weightSpan = unpackBuffer.AsSpan(0, inputDim);
                    var activationSpan = quantizedInput.AsSpan(activationOffset, inputDim);

                    var isum = UseSimd
                        ? TernaryDotSimd(weightSpan, activationSpan)
                        : TernaryDotScalar(weightSpan, activationSpan);

                    output[row, outputColumn] = isum * dequantScale;
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(unpackBuffer);
        }

        return output;
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);

        var rows = gradientOutput.GetLength(0);
        var outDim = Config.OutputDimension;
        var inDim = Config.InputDimension;
        var gradInput = new float[rows, inDim];

        var unpackBuffer = ArrayPool<sbyte>.Shared.Rent(inDim);
        try
        {
            for (var row = 0; row < rows; row++)
            {
                for (var outCol = 0; outCol < outDim; outCol++)
                {
                    var grad = gradientOutput[row, outCol] * Gamma;
                    if (grad == 0f)
                    {
                        continue;
                    }

                    var physicalRow = _rowPermutation is not null ? _rowPermutation[outCol] : outCol;
                    TritPacking.UnpackRowInto(_packedWeights, physicalRow * _packedStride, _packedStride, unpackBuffer, inDim);

                    for (var inCol = 0; inCol < inDim; inCol++)
                    {
                        var w = unpackBuffer[inCol];
                        if (w > 0) gradInput[row, inCol] += grad;
                        else if (w < 0) gradInput[row, inCol] -= grad;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(unpackBuffer);
        }

        if (_masterGradients is not null && _cachedInput is not null)
        {
            for (var row = 0; row < rows; row++)
            {
                for (var outCol = 0; outCol < outDim; outCol++)
                {
                    var grad = gradientOutput[row, outCol];
                    if (grad == 0f)
                    {
                        continue;
                    }

                    var weightOffset = outCol * inDim;
                    for (var inCol = 0; inCol < inDim; inCol++)
                    {
                        _masterGradients[weightOffset + inCol] += grad * _cachedInput[row, inCol];
                    }
                }
            }
        }

        return gradInput;
    }

    public void InitializeMasterWeights()
    {
        _masterWeights = new float[_totalWeights];
        _masterGradients = new float[_totalWeights];

        var inputDim = Config.InputDimension;
        var buffer = new sbyte[inputDim];
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, buffer, inputDim);
            var offset = row * inputDim;
            for (var col = 0; col < inputDim; col++)
            {
                _masterWeights[offset + col] = buffer[col] * Gamma;
            }
        }
    }

    public void ZeroGradients()
    {
        if (_masterGradients is not null)
        {
            Array.Clear(_masterGradients);
        }
    }

    public void SyncTernaryFromMaster()
    {
        if (_masterWeights is null)
        {
            return;
        }

        var absSum = 0f;
        for (var i = 0; i < _masterWeights.Length; i++)
        {
            absSum += MathF.Abs(_masterWeights[i]);
        }

        Gamma = _masterWeights.Length > 0 ? absSum / _masterWeights.Length : 0f;

        if (Gamma <= 0f)
        {
            Array.Clear(_packedWeights);
            return;
        }

        var ternary = new sbyte[_totalWeights];
        for (var i = 0; i < _masterWeights.Length; i++)
        {
            var normalized = _masterWeights[i] / Gamma + WeightQuantizationEpsilon;
            ternary[i] = (sbyte)Math.Clamp(
                (int)MathF.Round(normalized, MidpointRounding.AwayFromZero), -1, 1);
        }

        PackRowMajor(ternary);
    }

    public float[]? ExportMasterWeights() => _masterWeights is null ? null : [.. _masterWeights];

    public float[]? ExportMasterGradients() => _masterGradients is null ? null : [.. _masterGradients];

    public void ImportMasterWeights(float[] weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        if (weights.Length != _totalWeights)
        {
            throw new ArgumentException(
                $"Expected {_totalWeights} weights, got {weights.Length}.",
                nameof(weights));
        }

        _masterWeights ??= new float[weights.Length];
        _masterGradients ??= new float[weights.Length];
        weights.CopyTo(_masterWeights, 0);
    }

    public void ApplyRowPermutation(int[] permutation)
    {
        ArgumentNullException.ThrowIfNull(permutation);

        if (permutation.Length != Config.OutputDimension)
        {
            throw new ArgumentException(
                $"Permutation length {permutation.Length} does not match output dimension {Config.OutputDimension}.",
                nameof(permutation));
        }

        // Physically reorder packed weight rows
        var newPacked = new byte[_packedWeights.Length];
        for (var logical = 0; logical < Config.OutputDimension; logical++)
        {
            var physical = permutation[logical];
            Array.Copy(_packedWeights, logical * _packedStride, newPacked, physical * _packedStride, _packedStride);
        }

        _packedWeights = newPacked;
        _rowPermutation = (int[])permutation.Clone();
    }

    public int[]? ExportRowPermutation() => _rowPermutation is null ? null : (int[])_rowPermutation.Clone();

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
            Array.Clear(_packedWeights);
            return;
        }

        var inputDim = Config.InputDimension;
        var ternary = new sbyte[_totalWeights];
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            var offset = row * inputDim;
            for (var column = 0; column < inputDim; column++)
            {
                var normalized = fullPrecisionWeights[row, column] / Gamma;
                normalized += WeightQuantizationEpsilon;
                var quantized = Math.Clamp((int)MathF.Round(normalized, MidpointRounding.AwayFromZero), -1, 1);
                ternary[offset + column] = (sbyte)quantized;
            }
        }

        PackRowMajor(ternary);
    }

    public float[,] ToFullPrecision()
    {
        var result = new float[Config.OutputDimension, Config.InputDimension];
        var inputDim = Config.InputDimension;
        var buffer = new sbyte[inputDim];

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, buffer, inputDim);
            for (var column = 0; column < inputDim; column++)
            {
                result[row, column] = buffer[column] * Gamma;
            }
        }

        return result;
    }

    public TernaryWeightStats GetTernaryStats()
    {
        var negativeCount = 0;
        var zeroCount = 0;
        var positiveCount = 0;
        var inputDim = Config.InputDimension;
        var buffer = new sbyte[inputDim];

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            TritPacking.UnpackRowInto(_packedWeights, row * _packedStride, _packedStride, buffer, inputDim);
            for (var col = 0; col < inputDim; col++)
            {
                switch (buffer[col])
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
        }

        return new TernaryWeightStats(negativeCount, zeroCount, positiveCount);
    }

    private void PackRowMajor(sbyte[] ternary)
    {
        var inputDim = Config.InputDimension;
        for (var row = 0; row < Config.OutputDimension; row++)
        {
            var srcOffset = row * inputDim;
            var dstOffset = row * _packedStride;
            for (var pi = 0; pi < _packedStride; pi++)
            {
                var baseIdx = srcOffset + pi * 5;
                sbyte t0 = baseIdx < ternary.Length ? ternary[baseIdx] : (sbyte)0;
                sbyte t1 = baseIdx + 1 < ternary.Length ? ternary[baseIdx + 1] : (sbyte)0;
                sbyte t2 = baseIdx + 2 < ternary.Length ? ternary[baseIdx + 2] : (sbyte)0;
                sbyte t3 = baseIdx + 3 < ternary.Length ? ternary[baseIdx + 3] : (sbyte)0;
                sbyte t4 = baseIdx + 4 < ternary.Length ? ternary[baseIdx + 4] : (sbyte)0;
                _packedWeights[dstOffset + pi] = TritPacking.PackFive(t0, t1, t2, t3, t4);
            }
        }
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

            Vector.Widen(posVals, out var posLo, out var posHi);
            Vector.Widen(negVals, out var negLo, out var negHi);

            accumPos += posLo + posHi;
            accumNeg += negLo + negHi;
        }

        var result = 0;
        for (var j = 0; j < Vector<short>.Count; j++)
        {
            result += accumPos[j] - accumNeg[j];
        }

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
