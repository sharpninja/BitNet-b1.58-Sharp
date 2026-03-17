using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Utils;

internal static class ParameterInitializer
{
    public static BitLinear CreateBitLinear(BitLinearConfig config, Random random, float scale = 0.02f)
    {
        var layer = new BitLinear(config);
        layer.QuantizeFromFullPrecision(CreateMatrix(config.OutputDimension, config.InputDimension, random, scale));
        return layer;
    }

    public static float[,] CreateMatrix(int rows, int columns, Random random, float scale = 0.02f)
    {
        var values = new float[rows, columns];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                values[row, column] = ((float)random.NextDouble() * 2f - 1f) * scale;
            }
        }

        return values;
    }
}
