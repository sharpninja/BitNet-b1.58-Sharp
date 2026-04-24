namespace BitNetSharp.Core.Inference;

/// <summary>
/// IBERT-style softmax with an integer exp LUT. The table stores
/// exp(-k * delta) in Q16.16 for k in [0, lutEntries). Per row we find the
/// max, shift logits so every value is <= 0, index the LUT, then divide each
/// exp by the row sum in Q16.16. Values shifted below -maxShiftMagnitude
/// clamp to the last LUT entry (underflow to zero probability).
/// </summary>
public sealed class IntegerSoftmax
{
    private readonly int[] _expLut;
    private readonly int _lutEntries;
    private readonly float _maxShiftMagnitude;
    private readonly float _indexPerUnit;

    public IntegerSoftmax(int lutEntries = 4096, float maxShiftMagnitude = 32f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(lutEntries);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxShiftMagnitude);

        _lutEntries = lutEntries;
        _maxShiftMagnitude = maxShiftMagnitude;
        _indexPerUnit = lutEntries / maxShiftMagnitude;

        _expLut = new int[lutEntries];
        for (var i = 0; i < lutEntries; i++)
        {
            var x = -(i / (double)lutEntries) * maxShiftMagnitude;
            var e = Math.Exp(x) * IntegerMath.Q16_16_ONE;
            _expLut[i] = (int)Math.Clamp(Math.Round(e), 0, int.MaxValue);
        }
    }

    public float[,] ApplyToFloat(float[,] logits)
    {
        ArgumentNullException.ThrowIfNull(logits);
        var rows = logits.GetLength(0);
        var cols = logits.GetLength(1);
        var output = new float[rows, cols];
        var expBuf = new int[cols];

        for (var r = 0; r < rows; r++)
        {
            var max = float.NegativeInfinity;
            for (var c = 0; c < cols; c++)
            {
                if (logits[r, c] > max) max = logits[r, c];
            }

            long sum = 0;
            for (var c = 0; c < cols; c++)
            {
                var shifted = logits[r, c] - max;
                var floatIdx = -shifted * _indexPerUnit;
                if (floatIdx < 0f) floatIdx = 0f;
                var idx = (int)floatIdx;
                if (idx >= _lutEntries - 1)
                {
                    expBuf[c] = idx >= _lutEntries ? 0 : _expLut[_lutEntries - 1];
                }
                else
                {
                    var frac = floatIdx - idx;
                    var lo = _expLut[idx];
                    var hi = _expLut[idx + 1];
                    expBuf[c] = (int)(lo + (hi - lo) * frac);
                }
                sum += expBuf[c];
            }

            if (sum <= 0)
            {
                var uniform = 1f / cols;
                for (var c = 0; c < cols; c++) output[r, c] = uniform;
                continue;
            }

            for (var c = 0; c < cols; c++)
            {
                var probQ = ((long)expBuf[c] << IntegerMath.Q16_16_SHIFT) / sum;
                output[r, c] = IntegerMath.FromQ16_16(probQ);
            }
        }

        return output;
    }
}
