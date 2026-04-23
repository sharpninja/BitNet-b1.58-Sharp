using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Utils;

internal static class ParameterInitializer
{
    // Matrices above this element count go through a row-streaming seed path
    // that never allocates the full FP32 buffer, so large models (multi-B
    // parameters) can be constructed without exhausting the LOH.
    //
    // Threshold is set high enough (~32 MB FP32 per tensor) that small-model
    // training paths keep using the empirical-Gamma QuantizeFromFullPrecision
    // path (bit-exact with pre-streaming behavior). Only tensors that would
    // allocate >128 MB FP32 worth of transient buffer take the closed-form
    // Gamma streaming path, which trades a tiny Gamma/quantization mismatch
    // (~0.07% weight flips at 2M elements, vanishing as sqrt(n)) for bounded
    // O(row) peak memory.
    private const long StreamingThreshold = 8_000_000L;

    public static BitLinear CreateBitLinear(BitLinearConfig config, Random random, float scale = 0.02f)
    {
        var layer = new BitLinear(config);
        long elementCount = (long)config.OutputDimension * config.InputDimension;
        if (elementCount > StreamingThreshold)
        {
            SeedTernaryRowStream(layer, config, random, scale);
        }
        else
        {
            layer.QuantizeFromFullPrecision(CreateMatrix(config.OutputDimension, config.InputDimension, random, scale));
        }
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

    private static void SeedTernaryRowStream(BitLinear layer, BitLinearConfig config, Random random, float scale)
    {
        int rows = config.OutputDimension;
        int cols = config.InputDimension;
        long total = (long)rows * cols;

        // For a uniform [-scale, +scale] distribution, E[|w|] = scale / 2 exactly.
        // Use this closed-form Gamma so the streamed trit seed matches what
        // QuantizeFromFullPrecision would have picked on average.
        float gamma = scale * 0.5f;

        var trits = new sbyte[total];
        for (int r = 0; r < rows; r++)
        {
            long rowOffset = (long)r * cols;
            for (int c = 0; c < cols; c++)
            {
                float w = ((float)random.NextDouble() * 2f - 1f) * scale;
                // Ternary sign-based quantization: same rule as
                // BitLinear.QuantizeFromFullPrecision (clamp(round(w/gamma)) in [-1, +1]).
                float normalized = gamma > 0f ? w / gamma : 0f;
                int q = (int)MathF.Round(normalized, MidpointRounding.AwayFromZero);
                if (q < -1) q = -1; else if (q > 1) q = 1;
                trits[rowOffset + c] = (sbyte)q;
            }
        }
        layer.ImportTernary(trits, gamma);
    }
}
