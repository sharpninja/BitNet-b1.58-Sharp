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

    public void ApplyInPlace(float[,] tensor, int headCount)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);

        if (tensor.GetLength(1) != headCount * HeadDimension)
        {
            throw new ArgumentException("Tensor width must equal headCount * HeadDimension.", nameof(tensor));
        }

        var sequenceLength = tensor.GetLength(0);
        var halfDimension = HeadDimension / 2;

        // Precompute sin and cos tables for all positions and dimension pairs.
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

                    tensor[position, headOffset + dimension] = evenValue * cos - oddValue * sin;
                    tensor[position, headOffset + dimension + 1] = evenValue * sin + oddValue * cos;
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
