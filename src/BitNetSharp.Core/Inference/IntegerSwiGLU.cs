namespace BitNetSharp.Core.Inference;

/// <summary>
/// SwiGLU activation using a precomputed integer sigmoid LUT. For each
/// element output = gate * sigmoid(gate) * up. The LUT holds sigmoid(x)
/// as Q16.16 for x in [-maxMagnitude, +maxMagnitude] with linear interp;
/// inputs outside the range saturate to 0 or 1.
/// </summary>
public sealed class IntegerSwiGLU
{
    private readonly int[] _sigLut;
    private readonly int _lutEntries;
    private readonly float _maxMagnitude;
    private readonly float _indexPerUnit;

    public IntegerSwiGLU(int lutEntries = 4096, float maxMagnitude = 16f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(lutEntries);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxMagnitude);

        _lutEntries = lutEntries;
        _maxMagnitude = maxMagnitude;
        _indexPerUnit = (lutEntries - 1) / (2f * maxMagnitude);

        _sigLut = new int[lutEntries];
        for (var i = 0; i < lutEntries; i++)
        {
            var x = -maxMagnitude + (2.0 * maxMagnitude) * (i / (double)(lutEntries - 1));
            var s = 1.0 / (1.0 + Math.Exp(-x));
            _sigLut[i] = (int)Math.Round(s * IntegerMath.Q16_16_ONE);
        }
    }

    public float[,] ApplyToFloat(float[,] gate, float[,] up)
    {
        ArgumentNullException.ThrowIfNull(gate);
        ArgumentNullException.ThrowIfNull(up);
        if (gate.GetLength(0) != up.GetLength(0) || gate.GetLength(1) != up.GetLength(1))
        {
            throw new ArgumentException(
                "gate and up must have identical shape.",
                nameof(up));
        }

        var rows = gate.GetLength(0);
        var cols = gate.GetLength(1);
        var output = new float[rows, cols];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                var g = gate[r, c];
                int sigQ;
                if (g >= _maxMagnitude)
                {
                    sigQ = (int)IntegerMath.Q16_16_ONE;
                }
                else if (g <= -_maxMagnitude)
                {
                    sigQ = 0;
                }
                else
                {
                    var floatIdx = (g + _maxMagnitude) * _indexPerUnit;
                    var idx = (int)floatIdx;
                    if (idx >= _lutEntries - 1)
                    {
                        sigQ = _sigLut[_lutEntries - 1];
                    }
                    else
                    {
                        var frac = floatIdx - idx;
                        var lo = _sigLut[idx];
                        var hi = _sigLut[idx + 1];
                        sigQ = (int)(lo + (hi - lo) * frac);
                    }
                }

                var sigFloat = sigQ / (float)IntegerMath.Q16_16_ONE;
                output[r, c] = g * sigFloat * up[r, c];
            }
        }

        return output;
    }
}
