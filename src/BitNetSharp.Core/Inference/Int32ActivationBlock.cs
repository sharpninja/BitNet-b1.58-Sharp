namespace BitNetSharp.Core.Inference;

/// <summary>
/// Integer output block for BitLinear.ForwardInt32. Values are row-major
/// int32 accumulators (ternary dot products), and RowScales combine the
/// activation row scale with the layer Gamma into a single float per row.
/// Dequantisation: float v = Values[row, col] * RowScales[row].
/// </summary>
public sealed class Int32ActivationBlock
{
    public Int32ActivationBlock(int[,] values, float[] rowScales)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(rowScales);
        if (rowScales.Length != values.GetLength(0))
        {
            throw new ArgumentException(
                $"RowScales length {rowScales.Length} does not match values rows {values.GetLength(0)}.",
                nameof(rowScales));
        }

        Values = values;
        RowScales = rowScales;
    }

    public int[,] Values { get; }

    public float[] RowScales { get; }

    public int Rows => Values.GetLength(0);

    public int Cols => Values.GetLength(1);

    public float[,] ToFloat()
    {
        var rows = Rows;
        var cols = Cols;
        var output = new float[rows, cols];
        for (var row = 0; row < rows; row++)
        {
            var scale = RowScales[row];
            for (var col = 0; col < cols; col++)
            {
                output[row, col] = Values[row, col] * scale;
            }
        }
        return output;
    }
}
