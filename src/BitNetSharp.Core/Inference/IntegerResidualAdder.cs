namespace BitNetSharp.Core.Inference;

/// <summary>
/// Adds two <see cref="Int32ActivationBlock"/>s elementwise with per-row
/// scale alignment. For each row r the adder picks targetScale =
/// max(a.RowScales[r], b.RowScales[r]), rescales the smaller-scale row in
/// Q16.16 (int32 shift-round), then sums int values. The result carries
/// targetScale per row so later kernels see a single canonical scale.
/// </summary>
public sealed class IntegerResidualAdder
{
    public Int32ActivationBlock Add(Int32ActivationBlock a, Int32ActivationBlock b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Rows != b.Rows || a.Cols != b.Cols)
        {
            throw new ArgumentException(
                $"Shape mismatch: a is {a.Rows}x{a.Cols}, b is {b.Rows}x{b.Cols}.",
                nameof(b));
        }

        var rows = a.Rows;
        var cols = a.Cols;
        var sumValues = new int[rows, cols];
        var sumScales = new float[rows];

        for (var r = 0; r < rows; r++)
        {
            var scaleA = a.RowScales[r];
            var scaleB = b.RowScales[r];

            // Pick the smaller non-zero scale so both operands get upscaled
            // (preserving precision) rather than downscaled.
            float targetScale;
            if (scaleA == 0f) targetScale = scaleB;
            else if (scaleB == 0f) targetScale = scaleA;
            else targetScale = Math.Min(scaleA, scaleB);
            sumScales[r] = targetScale;

            if (targetScale == 0f)
            {
                for (var c = 0; c < cols; c++) sumValues[r, c] = 0;
                continue;
            }

            if (scaleA == scaleB)
            {
                for (var c = 0; c < cols; c++)
                {
                    sumValues[r, c] = a.Values[r, c] + b.Values[r, c];
                }
                continue;
            }

            var ratioA = (double)scaleA / targetScale;
            var ratioB = (double)scaleB / targetScale;

            for (var c = 0; c < cols; c++)
            {
                var av = (int)Math.Round(a.Values[r, c] * ratioA);
                var bv = (int)Math.Round(b.Values[r, c] * ratioB);
                sumValues[r, c] = av + bv;
            }
        }

        return new Int32ActivationBlock(sumValues, sumScales);
    }
}
