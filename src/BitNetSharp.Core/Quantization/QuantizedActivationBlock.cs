namespace BitNetSharp.Core.Quantization;

/// <summary>
/// Pre-computed absmax int8 quantization of an activation matrix. Produced
/// once per shared input (e.g. the norm output feeding Q/K/V) so the three
/// attention projections and the two feed-forward projections don't each
/// re-scan every row to find an absmax. Layout: <see cref="Quantized"/> is
/// row-major length <c>Rows*Cols</c>, <see cref="RowScales"/> is length Rows.
/// </summary>
public sealed class QuantizedActivationBlock
{
    private const int ActivationQuantizationMaxMagnitude = 127;

    // Async-local counter so phase-2 tests can assert a shared input is
    // quantised exactly once per attention / FFN block instead of once per
    // projection. Scoped per test to survive parallel test execution.
    internal static readonly System.Threading.AsyncLocal<StrongBox<long>> FromFloatCallCounter = new();

    internal sealed class StrongBox<T>
    {
        public T Value = default!;
    }

    public QuantizedActivationBlock(sbyte[] quantized, float[] rowScales, int rows, int cols)
    {
        ArgumentNullException.ThrowIfNull(quantized);
        ArgumentNullException.ThrowIfNull(rowScales);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(cols);
        if (quantized.Length != rows * cols)
        {
            throw new ArgumentException($"Quantized buffer length {quantized.Length} != rows*cols {rows * cols}.");
        }
        if (rowScales.Length != rows)
        {
            throw new ArgumentException($"RowScales length {rowScales.Length} != rows {rows}.");
        }

        Quantized = quantized;
        RowScales = rowScales;
        Rows = rows;
        Cols = cols;
    }

    public sbyte[] Quantized { get; }

    public float[] RowScales { get; }

    public int Rows { get; }

    public int Cols { get; }

    public static QuantizedActivationBlock FromFloat(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var counter = FromFloatCallCounter.Value;
        if (counter is not null)
        {
            counter.Value++;
        }

        var rows = input.GetLength(0);
        var cols = input.GetLength(1);
        var quantized = new sbyte[rows * cols];
        var rowScales = new float[rows];

        for (var row = 0; row < rows; row++)
        {
            var maxAbs = 0f;
            for (var column = 0; column < cols; column++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(input[row, column]));
            }

            if (maxAbs <= 0f)
            {
                rowScales[row] = 1f;
                continue;
            }

            var scale = maxAbs / ActivationQuantizationMaxMagnitude;
            rowScales[row] = scale;

            var offset = row * cols;
            for (var column = 0; column < cols; column++)
            {
                var q = (int)MathF.Round(input[row, column] / scale, MidpointRounding.AwayFromZero);
                quantized[offset + column] = (sbyte)Math.Clamp(q, -ActivationQuantizationMaxMagnitude, ActivationQuantizationMaxMagnitude);
            }
        }

        return new QuantizedActivationBlock(quantized, rowScales, rows, cols);
    }
}
