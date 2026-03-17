namespace BitNetSharp.Core.Utils;

public sealed class RotaryPositionEmbedding
{
    private readonly double _theta;

    public RotaryPositionEmbedding(int headDimension, double theta = 10_000d)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDimension);

        if (headDimension % 2 != 0)
        {
            throw new ArgumentException("Head dimension must be even for rotary embeddings.", nameof(headDimension));
        }

        HeadDimension = headDimension;
        _theta = theta;
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

        for (var position = 0; position < tensor.GetLength(0); position++)
        {
            for (var head = 0; head < headCount; head++)
            {
                var headOffset = head * HeadDimension;
                for (var dimension = 0; dimension < HeadDimension; dimension += 2)
                {
                    var pairIndex = dimension / 2d;
                    var angle = position / Math.Pow(_theta, (2d * pairIndex) / HeadDimension);
                    var cos = (float)Math.Cos(angle);
                    var sin = (float)Math.Sin(angle);

                    var evenValue = tensor[position, headOffset + dimension];
                    var oddValue = tensor[position, headOffset + dimension + 1];

                    tensor[position, headOffset + dimension] = evenValue * cos - oddValue * sin;
                    tensor[position, headOffset + dimension + 1] = evenValue * sin + oddValue * cos;
                }
            }
        }
    }
}
