using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Layers;

public sealed class BitLinear : Module
{
    private const int ActivationQuantizationMaxMagnitude = 127;
    private const float WeightQuantizationEpsilon = 1e-6f;

    private readonly sbyte[,] _ternaryWeights;

    public BitLinear(BitLinearConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        Config = config;
        _ternaryWeights = new sbyte[config.OutputDimension, config.InputDimension];
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

        if (input.GetLength(1) != Config.InputDimension)
        {
            throw new ArgumentException($"Expected input dimension {Config.InputDimension}, but received {input.GetLength(1)}.", nameof(input));
        }

        var (quantizedInput, rowScales) = QuantizeActivations(input);
        var rows = input.GetLength(0);
        var output = new float[rows, Config.OutputDimension];

        for (var row = 0; row < rows; row++)
        {
            var dequantScale = Gamma * rowScales[row];

            for (var outputColumn = 0; outputColumn < Config.OutputDimension; outputColumn++)
            {
                var isum = 0;
                for (var inputColumn = 0; inputColumn < Config.InputDimension; inputColumn++)
                {
                    var w = _ternaryWeights[outputColumn, inputColumn];
                    if (w > 0) isum += quantizedInput[row, inputColumn];
                    else if (w < 0) isum -= quantizedInput[row, inputColumn];
                }

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

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            for (var column = 0; column < Config.InputDimension; column++)
            {
                var normalized = fullPrecisionWeights[row, column] / Gamma;
                normalized += WeightQuantizationEpsilon;
                var quantized = Math.Clamp((int)MathF.Round(normalized, MidpointRounding.AwayFromZero), -1, 1);
                _ternaryWeights[row, column] = (sbyte)quantized;
            }
        }
    }

    public float[,] ToFullPrecision()
    {
        var result = new float[Config.OutputDimension, Config.InputDimension];

        for (var row = 0; row < Config.OutputDimension; row++)
        {
            for (var column = 0; column < Config.InputDimension; column++)
            {
                result[row, column] = _ternaryWeights[row, column] * Gamma;
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

    private static (sbyte[,] quantized, float[] rowScales) QuantizeActivations(float[,] input)
    {
        var rows = input.GetLength(0);
        var cols = input.GetLength(1);
        var quantized = new sbyte[rows, cols];
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

            for (var column = 0; column < cols; column++)
            {
                var q = (int)MathF.Round(input[row, column] / scale, MidpointRounding.AwayFromZero);
                quantized[row, column] = (sbyte)Math.Clamp(q, -ActivationQuantizationMaxMagnitude, ActivationQuantizationMaxMagnitude);
            }
        }

        return (quantized, rowScales);
    }
}
