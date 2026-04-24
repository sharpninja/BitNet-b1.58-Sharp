namespace BitNetSharp.Core.Utils;

public sealed class RotaryPositionEmbedding
{
    private readonly double _theta;
    private readonly double[] _inverseFrequencies;

    public RotaryPositionEmbedding(int headDimension, double theta = 10_000d)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

        if (headDimension % 2 != 0)
        {
            throw new ArgumentException("Head dimension must be even for rotary embeddings.", nameof(headDimension));
        }

        HeadDimension = headDimension;
        _theta = theta;

        var halfDimension = HeadDimension / 2;
        _inverseFrequencies = new double[halfDimension];
        for (var pairIndex = 0; pairIndex < halfDimension; pairIndex++)
        {
            var exponent = (2d * pairIndex) / HeadDimension;
            _inverseFrequencies[pairIndex] = 1d / Math.Pow(_theta, exponent);
        }
    }

    public int HeadDimension { get; }

    public void ApplyInPlace(float[,] tensor, int headCount) => ApplyInPlace(tensor, headCount, positionOffset: 0);

    /// <summary>
    /// Apply RoPE in-place treating row <c>r</c> as sequence position
    /// <c>positionOffset + r</c>. Used at decode time so the single new row is
    /// rotated by its true absolute position without rebuilding the sin/cos
    /// table for every prior position.
    /// </summary>
    public void ApplyInPlace(float[,] tensor, int headCount, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);

        if (tensor.GetLength(1) != headCount * HeadDimension)
        {
            throw new ArgumentException("Tensor width must equal headCount * HeadDimension.", nameof(tensor));
        }

        var sequenceLength = tensor.GetLength(0);
        var halfDimension = HeadDimension / 2;

        for (var row = 0; row < sequenceLength; row++)
        {
            var absolutePosition = positionOffset + row;
            for (var head = 0; head < headCount; head++)
            {
                var headOffset = head * HeadDimension;
                for (var pairIndex = 0; pairIndex < halfDimension; pairIndex++)
                {
                    var dimension = pairIndex * 2;
                    var angle = absolutePosition * _inverseFrequencies[pairIndex];
                    var cos = (float)Math.Cos(angle);
                    var sin = (float)Math.Sin(angle);

                    var evenValue = tensor[row, headOffset + dimension];
                    var oddValue = tensor[row, headOffset + dimension + 1];

                    tensor[row, headOffset + dimension] = evenValue * cos - oddValue * sin;
                    tensor[row, headOffset + dimension + 1] = evenValue * sin + oddValue * cos;
                }
            }
        }
    }

    public void ApplyInverseInPlace(float[,] tensor, int headCount)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);

        if (tensor.GetLength(1) != headCount * HeadDimension)
        {
            throw new ArgumentException("Tensor width must equal headCount * HeadDimension.", nameof(tensor));
        }

        var sequenceLength = tensor.GetLength(0);
        var halfDimension = HeadDimension / 2;

        var cosTable = new float[sequenceLength, halfDimension];
        var sinTable = new float[sequenceLength, halfDimension];

        for (var position = 0; position < sequenceLength; position++)
        {
            for (var pairIndex = 0; pairIndex < halfDimension; pairIndex++)
            {
                var angle = position * _inverseFrequencies[pairIndex];
                cosTable[position, pairIndex] = (float)Math.Cos(angle);
                sinTable[position, pairIndex] = (float)Math.Sin(angle);
            }
        }

        // Inverse rotation: transpose of [cos, -sin; sin, cos] is [cos, sin; -sin, cos]
        for (var position = 0; position < sequenceLength; position++)
        {
            for (var head = 0; head < headCount; head++)
            {
                var headOffset = head * HeadDimension;
                for (var pairIndex = 0; pairIndex < halfDimension; pairIndex++)
                {
                    var dimension = pairIndex * 2;
                    var cos = cosTable[position, pairIndex];
                    var sin = sinTable[position, pairIndex];

                    var evenValue = tensor[position, headOffset + dimension];
                    var oddValue = tensor[position, headOffset + dimension + 1];

                    tensor[position, headOffset + dimension] = evenValue * cos + oddValue * sin;
                    tensor[position, headOffset + dimension + 1] = -evenValue * sin + oddValue * cos;
                }
            }
        }
    }
}
