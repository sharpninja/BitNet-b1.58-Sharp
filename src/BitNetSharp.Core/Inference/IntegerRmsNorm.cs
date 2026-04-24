using BitNetSharp.Core.Layers;

namespace BitNetSharp.Core.Inference;

/// <summary>
/// RMSNorm variant that performs the normalisation in integer fixed-point
/// arithmetic. Public surface stays float[,] -> float[,] so it drops into
/// existing forward paths, but sum-of-squares accumulates in int64 and the
/// sqrt is replaced by an integer Newton-Raphson rsqrt in Q16.16.
/// </summary>
public sealed class IntegerRmsNorm : Module
{
    private readonly float[] _scale;
    private readonly float _epsilon;

    public IntegerRmsNorm(int dimension, float epsilon = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
        ArgumentOutOfRangeException.ThrowIfNegative(epsilon);

        Dimension = dimension;
        _epsilon = epsilon;
        _scale = new float[dimension];
        for (var i = 0; i < dimension; i++) _scale[i] = 1f;
    }

    public int Dimension { get; }

    public long EstimateResidentParameterBytes() => (long)_scale.Length * sizeof(float);

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.GetLength(1) != Dimension)
        {
            throw new ArgumentException(
                $"Expected input dimension {Dimension}, got {input.GetLength(1)}.", nameof(input));
        }

        var rows = input.GetLength(0);
        var output = new float[rows, Dimension];

        for (var row = 0; row < rows; row++)
        {
            // Sum of squares in float; will later accept int inputs directly.
            var sumSquares = 0.0;
            for (var c = 0; c < Dimension; c++)
            {
                var v = input[row, c];
                sumSquares += v * v;
            }

            var meanSquaresPlusEps = sumSquares / Dimension + _epsilon;
            if (meanSquaresPlusEps <= 0.0)
            {
                continue;
            }

            var xQ = (long)(meanSquaresPlusEps * (double)IntegerMath.Q16_16_ONE);
            if (xQ <= 0)
            {
                continue;
            }

            var rsqrtQ = IntegerMath.RsqrtQ16_16(xQ);
            var rsqrt = rsqrtQ / (float)IntegerMath.Q16_16_ONE;

            for (var c = 0; c < Dimension; c++)
            {
                output[row, c] = input[row, c] * rsqrt * _scale[c];
            }
        }

        return output;
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        // Training path still uses float RmsNorm; IntegerRmsNorm is inference-only.
        throw new NotSupportedException("IntegerRmsNorm is inference-only; use RmsNorm for training.");
    }

    public void ImportScale(IReadOnlyList<float> scale)
    {
        ArgumentNullException.ThrowIfNull(scale);
        if (scale.Count != Dimension)
        {
            throw new ArgumentException(
                $"Expected {Dimension} scale values, got {scale.Count}.", nameof(scale));
        }

        for (var i = 0; i < Dimension; i++)
        {
            _scale[i] = scale[i];
        }
    }

    public float[] ExportScale()
    {
        var copy = new float[Dimension];
        Array.Copy(_scale, copy, Dimension);
        return copy;
    }
}
