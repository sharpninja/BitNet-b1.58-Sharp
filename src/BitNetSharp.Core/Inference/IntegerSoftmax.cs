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
        Span<int> expBuf = cols <= 256 ? stackalloc int[cols] : new int[cols];
        Span<float> rowBuf = cols <= 256 ? stackalloc float[cols] : new float[cols];

        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++) rowBuf[c] = logits[r, c];
            ApplyCore(rowBuf, rowBuf, expBuf);
            for (var c = 0; c < cols; c++) output[r, c] = rowBuf[c];
        }

        return output;
    }

    /// <summary>
    /// Phase F7: per-row softmax that writes into a caller-supplied output
    /// span instead of allocating a fresh float[,] per call. <paramref name="logits"/>
    /// and <paramref name="output"/> may alias the same span. Matches
    /// <see cref="ApplyToFloat"/> bit-for-bit on the same input row.
    /// </summary>
    public void ApplyRowInPlace(ReadOnlySpan<float> logits, Span<float> output)
    {
        if (logits.Length != output.Length)
        {
            throw new ArgumentException(
                $"Output length {output.Length} does not match logits length {logits.Length}.",
                nameof(output));
        }

        var cols = logits.Length;
        if (cols == 0)
        {
            return;
        }

        Span<int> expBuf = cols <= 256 ? stackalloc int[cols] : new int[cols];
        ApplyCore(logits, output, expBuf);
    }

    private void ApplyCore(ReadOnlySpan<float> logits, Span<float> output, Span<int> expBuf)
    {
        var cols = logits.Length;
        var max = float.NegativeInfinity;
        for (var c = 0; c < cols; c++)
        {
            if (logits[c] > max) max = logits[c];
        }

        long sum = 0;
        for (var c = 0; c < cols; c++)
        {
            var shifted = logits[c] - max;
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
            for (var c = 0; c < cols; c++) output[c] = uniform;
            return;
        }

        for (var c = 0; c < cols; c++)
        {
            var probQ = ((long)expBuf[c] << IntegerMath.Q16_16_SHIFT) / sum;
            output[c] = IntegerMath.FromQ16_16(probQ);
        }
    }
}
