namespace BitNetSharp.Core.Inference;

/// <summary>
/// Paged integer-backed K/V cache slab for one transformer layer. Rows
/// partition into a body (int8 values, one float scale per row) for the bulk
/// of the context and a tail (int32 values, one float scale per row) for the
/// most recent tokens, where precision matters most for softmax dynamics.
///
/// The class stores values as integers so the cache has a 4-8x memory
/// footprint reduction vs the float[,] backing it replaces, and downstream
/// integer attention kernels can avoid the dequant round-trip entirely.
/// Float scales are temporarily retained to keep the API compatible with the
/// existing float-attention reader; later I-stages pivot to int16 Q-scales.
/// </summary>
public sealed class IntegerKvCacheLayer
{
    private readonly sbyte[,] _kBody;
    private readonly sbyte[,] _vBody;
    private readonly int[,] _kTail;
    private readonly int[,] _vTail;
    private readonly float[] _kBodyScales;
    private readonly float[] _vBodyScales;
    private readonly float[] _kTailScales;
    private readonly float[] _vTailScales;
    private readonly int _tailFirstRow;

    public IntegerKvCacheLayer(int capacity, int kvDimension, int tailRows)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDimension);
        ArgumentOutOfRangeException.ThrowIfNegative(tailRows);
        if (tailRows > capacity)
        {
            throw new ArgumentOutOfRangeException(nameof(tailRows),
                $"tailRows {tailRows} cannot exceed capacity {capacity}.");
        }

        Capacity = capacity;
        KvDimension = kvDimension;
        TailRows = tailRows;
        _tailFirstRow = capacity - tailRows;

        _kBody = new sbyte[capacity, kvDimension];
        _vBody = new sbyte[capacity, kvDimension];
        _kTail = new int[capacity, kvDimension];
        _vTail = new int[capacity, kvDimension];
        _kBodyScales = new float[capacity];
        _vBodyScales = new float[capacity];
        _kTailScales = new float[capacity];
        _vTailScales = new float[capacity];
    }

    public int Capacity { get; }

    public int KvDimension { get; }

    public int TailRows { get; }

    private bool IsTail(int row) => row >= _tailFirstRow;

    public void WriteKRow(int row, ReadOnlySpan<float> values)
    {
        ValidateRow(row);
        ValidateLength(values.Length);
        if (IsTail(row))
        {
            WriteTailRow(_kTail, _kTailScales, row, values);
        }
        else
        {
            WriteBodyRow(_kBody, _kBodyScales, row, values);
        }
    }

    public void WriteVRow(int row, ReadOnlySpan<float> values)
    {
        ValidateRow(row);
        ValidateLength(values.Length);
        if (IsTail(row))
        {
            WriteTailRow(_vTail, _vTailScales, row, values);
        }
        else
        {
            WriteBodyRow(_vBody, _vBodyScales, row, values);
        }
    }

    public void ReadKRow(int row, Span<float> destination)
    {
        ValidateRow(row);
        ValidateLength(destination.Length);
        if (IsTail(row))
        {
            ReadTailRow(_kTail, _kTailScales, row, destination);
        }
        else
        {
            ReadBodyRow(_kBody, _kBodyScales, row, destination);
        }
    }

    public void ReadVRow(int row, Span<float> destination)
    {
        ValidateRow(row);
        ValidateLength(destination.Length);
        if (IsTail(row))
        {
            ReadTailRow(_vTail, _vTailScales, row, destination);
        }
        else
        {
            ReadBodyRow(_vBody, _vBodyScales, row, destination);
        }
    }

    public void Reset()
    {
        Array.Clear(_kBody);
        Array.Clear(_vBody);
        Array.Clear(_kTail);
        Array.Clear(_vTail);
        Array.Clear(_kBodyScales);
        Array.Clear(_vBodyScales);
        Array.Clear(_kTailScales);
        Array.Clear(_vTailScales);
    }

    private void WriteBodyRow(sbyte[,] body, float[] scales, int row, ReadOnlySpan<float> values)
    {
        var maxAbs = 0f;
        for (var i = 0; i < values.Length; i++)
        {
            var abs = MathF.Abs(values[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        if (maxAbs == 0f)
        {
            scales[row] = 0f;
            for (var i = 0; i < values.Length; i++)
            {
                body[row, i] = 0;
            }
            return;
        }

        var scale = maxAbs / 127f;
        var inverseScale = 1f / scale;
        scales[row] = scale;
        for (var i = 0; i < values.Length; i++)
        {
            var q = (int)MathF.Round(values[i] * inverseScale);
            body[row, i] = (sbyte)Math.Clamp(q, -127, 127);
        }
    }

    private void WriteTailRow(int[,] tail, float[] scales, int row, ReadOnlySpan<float> values)
    {
        var maxAbs = 0f;
        for (var i = 0; i < values.Length; i++)
        {
            var abs = MathF.Abs(values[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        if (maxAbs == 0f)
        {
            scales[row] = 0f;
            for (var i = 0; i < values.Length; i++)
            {
                tail[row, i] = 0;
            }
            return;
        }

        // Int32 range keeps tail quantisation essentially lossless relative to
        // float input: scale picks ~2^30 units as the max-magnitude slot so
        // fractional information survives the round trip.
        var scale = maxAbs / 1073741824f; // 2^30
        var inverseScale = 1f / scale;
        scales[row] = scale;
        for (var i = 0; i < values.Length; i++)
        {
            var q = (long)MathF.Round(values[i] * inverseScale);
            tail[row, i] = (int)Math.Clamp(q, int.MinValue, int.MaxValue);
        }
    }

    private static void ReadBodyRow(sbyte[,] body, float[] scales, int row, Span<float> destination)
    {
        var scale = scales[row];
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = body[row, i] * scale;
        }
    }

    private static void ReadTailRow(int[,] tail, float[] scales, int row, Span<float> destination)
    {
        var scale = scales[row];
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = tail[row, i] * scale;
        }
    }

    private void ValidateRow(int row)
    {
        if ((uint)row >= (uint)Capacity)
        {
            throw new ArgumentOutOfRangeException(nameof(row), row,
                $"Row {row} is outside [0, {Capacity}).");
        }
    }

    private void ValidateLength(int length)
    {
        if (length != KvDimension)
        {
            throw new ArgumentException(
                $"Expected row length {KvDimension}, got {length}.", nameof(length));
        }
    }
}
