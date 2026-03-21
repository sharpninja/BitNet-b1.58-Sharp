using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Layers;

public sealed class BitLinear : Module
{
    private const int ActivationQuantizationMaxMagnitude = 127;
    private const float WeightQuantizationEpsilon = 1e-6f;

    private readonly float[,] _fullPrecisionWeights;
    private readonly sbyte[,] _ternaryWeights;

    public BitLinear(BitLinearConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        Config = config;
        _fullPrecisionWeights = new float[config.OutputDimension, config.InputDimension];
        _ternaryWeights = new sbyte[config.OutputDimension, config.InputDimension];
    }

    public BitLinearConfig Config { get; }

    public float Gamma { get; private set; }

    public bool HasBias => false;

    public int ActivationQuantizationBound => ActivationQuantizationMaxMagnitude;

    public int ActivationQuantizationBitWidth => 8;

    public long EstimateResidentParameterBytes() =>
        ((long)_fullPrecisionWeights.Length * sizeof(float)) + ((long)_ternaryWeights.Length * sizeof(sbyte));

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.GetLength(1) != Config.InputDimension)
        {
            throw new ArgumentException($"Expected input dimension {Config.InputDimension}, but received {input.GetLength(1)}.", nameof(input));
        }

        var quantizedInput = QuantizeActivations(input);
        var output = new float[input.GetLength(0), Config.OutputDimension];

        for (var row = 0; row < quantizedInput.GetLength(0); row++)
        {
            for (var outputColumn = 0; outputColumn < Config.OutputDimension; outputColumn++)
            {
                var sum = 0f;
                for (var inputColumn = 0; inputColumn < Config.InputDimension; inputColumn++)
                {
                    sum += quantizedInput[row, inputColumn] * _ternaryWeights[outputColumn, inputColumn];
                }

                output[row, outputColumn] = sum * Gamma;
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

        Buffer.BlockCopy(fullPrecisionWeights, 0, _fullPrecisionWeights, 0, sizeof(float) * fullPrecisionWeights.Length);
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

    private static float[,] QuantizeActivations(float[,] input)
    {
        var result = new float[input.GetLength(0), input.GetLength(1)];

        for (var row = 0; row < input.GetLength(0); row++)
        {
            var maxAbs = 0f;
            for (var column = 0; column < input.GetLength(1); column++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(input[row, column]));
            }

            if (maxAbs <= 0f)
            {
                continue;
            }

            var scale = maxAbs / ActivationQuantizationMaxMagnitude;
            for (var column = 0; column < input.GetLength(1); column++)
            {
                var quantized = Math.Clamp((int)MathF.Round(input[row, column] / scale, MidpointRounding.AwayFromZero), -ActivationQuantizationMaxMagnitude, ActivationQuantizationMaxMagnitude);
                result[row, column] = quantized * scale;
            }
        }

        return result;
    }
}
