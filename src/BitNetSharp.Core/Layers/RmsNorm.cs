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

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.GetLength(1) != Dimension)
        {
            throw new ArgumentException($"Expected input dimension {Dimension}, but received {input.GetLength(1)}.", nameof(input));
        }

        var output = new float[input.GetLength(0), Dimension];

        for (var row = 0; row < input.GetLength(0); row++)
        {
            var sumSquares = 0f;
            for (var column = 0; column < Dimension; column++)
            {
                sumSquares += input[row, column] * input[row, column];
            }

            var rms = MathF.Sqrt(sumSquares / Dimension + Epsilon);
            for (var column = 0; column < Dimension; column++)
            {
                output[row, column] = input[row, column] / rms * _scale[column];
            }
        }

        return output;
    }
}
