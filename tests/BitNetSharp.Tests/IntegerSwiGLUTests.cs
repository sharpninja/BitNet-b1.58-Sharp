using BitNetSharp.Core.Inference;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I7: SwiGLU activation via an integer sigmoid LUT. For each element
/// output = gate * sigmoid(gate) * up. The LUT stores sigmoid(x) in Q16.16
/// over x in [-maxMagnitude, +maxMagnitude]; saturated endpoints clamp to 0
/// and 1. Result tracks the float reference within 1e-3 per element.
/// </summary>
public sealed class IntegerSwiGLUTests
{
    [Fact]
    public void ApplyToFloat_MatchesReferenceSwiGLU_WithinTolerance()
    {
        const int rows = 3;
        const int cols = 16;
        var rng = new Random(23);
        var gate = new float[rows, cols];
        var up = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                gate[r, c] = ((float)rng.NextDouble() - 0.5f) * 8f;
                up[r, c] = ((float)rng.NextDouble() - 0.5f) * 4f;
            }
        }

        var expected = ReferenceSwiGLU(gate, up);
        var integer = new IntegerSwiGLU();
        var actual = integer.ApplyToFloat(gate, up);

        Assert.Equal(rows, actual.GetLength(0));
        Assert.Equal(cols, actual.GetLength(1));
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                Assert.InRange(actual[r, c] - expected[r, c], -1e-3f, 1e-3f);
            }
        }
    }

    [Fact]
    public void ApplyToFloat_LargePositiveGate_SaturatesToGateTimesUp()
    {
        var gate = new float[1, 2] { { 50f, 30f } };
        var up = new float[1, 2] { { 1.5f, -2f } };
        var integer = new IntegerSwiGLU();
        var actual = integer.ApplyToFloat(gate, up);

        // sigmoid(50) ~ 1, so result ~ gate * up
        Assert.InRange(actual[0, 0] - (50f * 1.5f), -1e-2f, 1e-2f);
        Assert.InRange(actual[0, 1] - (30f * -2f), -1e-2f, 1e-2f);
    }

    [Fact]
    public void ApplyToFloat_LargeNegativeGate_SaturatesToZero()
    {
        var gate = new float[1, 2] { { -50f, -30f } };
        var up = new float[1, 2] { { 5f, -7f } };
        var integer = new IntegerSwiGLU();
        var actual = integer.ApplyToFloat(gate, up);

        Assert.InRange(actual[0, 0], -1e-3f, 1e-3f);
        Assert.InRange(actual[0, 1], -1e-3f, 1e-3f);
    }

    [Fact]
    public void ApplyToFloat_RejectsShapeMismatch()
    {
        var gate = new float[2, 3];
        var up = new float[2, 4];
        var integer = new IntegerSwiGLU();

        Assert.Throws<ArgumentException>(() => integer.ApplyToFloat(gate, up));
    }

    [Fact]
    public void Ctor_RejectsInvalidLutSize()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IntegerSwiGLU(lutEntries: 0, maxMagnitude: 16f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IntegerSwiGLU(lutEntries: 1024, maxMagnitude: 0f));
    }

    private static float[,] ReferenceSwiGLU(float[,] gate, float[,] up)
    {
        var rows = gate.GetLength(0);
        var cols = gate.GetLength(1);
        var output = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                var g = gate[r, c];
                var sig = 1.0 / (1.0 + Math.Exp(-g));
                output[r, c] = (float)(g * sig * up[r, c]);
            }
        }
        return output;
    }
}
