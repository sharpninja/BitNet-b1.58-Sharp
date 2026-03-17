namespace BitNetSharp.Core.Utils;

internal static class TensorMath
{
    public static float[,] Add(float[,] left, float[,] right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.GetLength(0) != right.GetLength(0) || left.GetLength(1) != right.GetLength(1))
        {
            throw new ArgumentException("Tensor shapes must match.");
        }

        var result = new float[left.GetLength(0), left.GetLength(1)];

        for (var row = 0; row < left.GetLength(0); row++)
        {
            for (var column = 0; column < left.GetLength(1); column++)
            {
                result[row, column] = left[row, column] + right[row, column];
            }
        }

        return result;
    }

    public static float[,] ElementwiseMultiply(float[,] left, float[,] right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left.GetLength(0) != right.GetLength(0) || left.GetLength(1) != right.GetLength(1))
        {
            throw new ArgumentException("Tensor shapes must match.");
        }

        var result = new float[left.GetLength(0), left.GetLength(1)];

        for (var row = 0; row < left.GetLength(0); row++)
        {
            for (var column = 0; column < left.GetLength(1); column++)
            {
                result[row, column] = left[row, column] * right[row, column];
            }
        }

        return result;
    }
}
