namespace BitNetSharp.Core.Layers;

public sealed class RmsNorm : Module
{
    private readonly float[] _scale;

    public RmsNorm(int dimension, float epsilon = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
        ArgumentOutOfRangeException.ThrowIfNegative(epsilon);

        Dimension = dimension;
        Epsilon = epsilon;
        _scale = Enumerable.Repeat(1f, dimension).ToArray();
    }

    public int Dimension { get; }

    public float Epsilon { get; }

    public bool HasBias => false;

    public bool HasLearnableScale => true;

    public long EstimateResidentParameterBytes() => (long)_scale.Length * sizeof(float);

    // Cached for backward pass
    private float[,]? _cachedInput;
    private float[]? _cachedRms;

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.GetLength(1) != Dimension)
        {
            throw new ArgumentException($"Expected input dimension {Dimension}, but received {input.GetLength(1)}.", nameof(input));
        }

        var rows = input.GetLength(0);
        var output = new float[rows, Dimension];
        var rmsValues = new float[rows];

        for (var row = 0; row < rows; row++)
        {
            var sumSquares = 0f;
            for (var column = 0; column < Dimension; column++)
            {
                sumSquares += input[row, column] * input[row, column];
            }

            var rms = MathF.Sqrt(sumSquares / Dimension + Epsilon);
            rmsValues[row] = rms;
            for (var column = 0; column < Dimension; column++)
            {
                output[row, column] = input[row, column] / rms * _scale[column];
            }
        }

        _cachedInput = (float[,])input.Clone();
        _cachedRms = rmsValues;

        return output;
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);

        if (_cachedInput is null || _cachedRms is null)
        {
            return (float[,])gradientOutput.Clone();
        }

        var rows = gradientOutput.GetLength(0);
        var gradInput = new float[rows, Dimension];

        for (var row = 0; row < rows; row++)
        {
            var rms = _cachedRms[row];
            if (rms <= 0f)
            {
                continue;
            }

            var invRms = 1f / rms;

            // Compute dot(gradOutput * scale, normalized_input) for the correction term
            var dotProduct = 0f;
            for (var col = 0; col < Dimension; col++)
            {
                var normalized = _cachedInput[row, col] * invRms;
                dotProduct += gradientOutput[row, col] * _scale[col] * normalized;
            }

            var correction = dotProduct / (rms * rms * Dimension);

            for (var col = 0; col < Dimension; col++)
            {
                // dL/dInput = (gradOutput * scale / rms) - input * correction
                gradInput[row, col] = gradientOutput[row, col] * _scale[col] * invRms
                    - _cachedInput[row, col] * correction;
            }
        }

        return gradInput;
    }

    internal float[] ExportScale() => [.. _scale];

    internal void ImportScale(IReadOnlyList<float> scale)
    {
        ArgumentNullException.ThrowIfNull(scale);

        if (scale.Count != Dimension)
        {
            throw new ArgumentException($"Expected {Dimension} scale values, but received {scale.Count}.", nameof(scale));
        }

        for (var index = 0; index < Dimension; index++)
        {
            _scale[index] = scale[index];
        }
    }
}
