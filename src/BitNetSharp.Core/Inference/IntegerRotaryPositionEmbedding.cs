namespace BitNetSharp.Core.Inference;

/// <summary>
/// RoPE variant whose sin/cos table is precomputed as int16 Q1.15 at
/// construction and applied via integer multiply-add at runtime. Q1.15 packs
/// values in [-1, 1) into a signed 16-bit slot, so the rotation kernel is
/// one int16 multiply + one int16 multiply + a shift per pair.
/// </summary>
public sealed class IntegerRotaryPositionEmbedding
{
    private const int Q1_15_SHIFT = 15;
    private const float Q1_15_ONE = 32768f;

    private readonly short[] _sinTable;
    private readonly short[] _cosTable;

    public IntegerRotaryPositionEmbedding(int headDim, int maxSequenceLength, double theta = 10_000d)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);
        if (headDim % 2 != 0)
        {
            throw new ArgumentException("Head dimension must be even.", nameof(headDim));
        }

        HeadDimension = headDim;
        MaxSequenceLength = maxSequenceLength;
        var halfDim = headDim / 2;
        _sinTable = new short[maxSequenceLength * halfDim];
        _cosTable = new short[maxSequenceLength * halfDim];

        for (var pos = 0; pos < maxSequenceLength; pos++)
        {
            for (var pair = 0; pair < halfDim; pair++)
            {
                var exponent = (2d * pair) / headDim;
                var invFreq = 1d / Math.Pow(theta, exponent);
                var angle = pos * invFreq;
                var sin = Math.Sin(angle);
                var cos = Math.Cos(angle);

                _sinTable[pos * halfDim + pair] = ToQ1_15(sin);
                _cosTable[pos * halfDim + pair] = ToQ1_15(cos);
            }
        }
    }

    public int HeadDimension { get; }

    public int MaxSequenceLength { get; }

    public void ApplyInPlace(float[,] tensor, int headCount) =>
        ApplyInPlace(tensor, headCount, positionOffset: 0);

    public void ApplyInPlace(float[,] tensor, int headCount, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegative(positionOffset);

        if (tensor.GetLength(1) != headCount * HeadDimension)
        {
            throw new ArgumentException("Tensor width must equal headCount * HeadDimension.", nameof(tensor));
        }

        var rows = tensor.GetLength(0);
        if (positionOffset + rows > MaxSequenceLength)
        {
            throw new ArgumentOutOfRangeException(nameof(positionOffset),
                $"positionOffset ({positionOffset}) + rows ({rows}) > maxSequenceLength ({MaxSequenceLength}).");
        }

        var halfDim = HeadDimension / 2;
        for (var row = 0; row < rows; row++)
        {
            var absolutePosition = positionOffset + row;
            var tableOffset = absolutePosition * halfDim;

            for (var head = 0; head < headCount; head++)
            {
                var headOffset = head * HeadDimension;
                for (var pair = 0; pair < halfDim; pair++)
                {
                    var dimension = pair * 2;
                    var cosQ = _cosTable[tableOffset + pair];
                    var sinQ = _sinTable[tableOffset + pair];

                    var even = tensor[row, headOffset + dimension];
                    var odd = tensor[row, headOffset + dimension + 1];

                    // Q1.15 * float -> float: divide by 32768 post-mul.
                    var newEven = (even * cosQ - odd * sinQ) / Q1_15_ONE;
                    var newOdd = (even * sinQ + odd * cosQ) / Q1_15_ONE;

                    tensor[row, headOffset + dimension] = newEven;
                    tensor[row, headOffset + dimension + 1] = newOdd;
                }
            }
        }
    }

    public (short[] Sin, short[] Cos) ExportSinCosTable()
    {
        var sin = new short[_sinTable.Length];
        var cos = new short[_cosTable.Length];
        Array.Copy(_sinTable, sin, _sinTable.Length);
        Array.Copy(_cosTable, cos, _cosTable.Length);
        return (sin, cos);
    }

    private static short ToQ1_15(double value)
    {
        // Q1.15: range [-1, 1). Clamp to [-32768, 32767] and round.
        var scaled = Math.Round(value * Q1_15_ONE);
        return (short)Math.Clamp(scaled, -32768.0, 32767.0);
    }
}
