using BitNetSharp.Core.Inference;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I8: residual add with per-row scale alignment. Two
/// <see cref="Int32ActivationBlock"/> inputs may carry different row scales,
/// so before summing integer values each row is rescaled to the larger of
/// the two scales via Q16.16 ratio. The resulting block has the larger
/// scale and int32 values representing the float sum within 1e-4.
/// </summary>
public sealed class IntegerResidualAdderTests
{
    [Fact]
    public void Add_MatchesFloatSum_WithinTolerance()
    {
        const int rows = 3;
        const int cols = 8;
        var rng = new Random(29);
        var a = MakeBlock(rows, cols, rng, scaleBase: 1e-3f);
        var b = MakeBlock(rows, cols, rng, scaleBase: 3e-3f);

        var adder = new IntegerResidualAdder();
        var sum = adder.Add(a, b);

        var aFloat = a.ToFloat();
        var bFloat = b.ToFloat();
        var expected = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                expected[r, c] = aFloat[r, c] + bFloat[r, c];
            }
        }

        var actualFloat = sum.ToFloat();
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                Assert.InRange(actualFloat[r, c] - expected[r, c], -1e-4f, 1e-4f);
            }
        }
    }

    [Fact]
    public void Add_SameScalesRowForRow_DoesNotRescale()
    {
        var values = new int[1, 4] { { 10, -5, 3, 0 } };
        var scale = new[] { 0.25f };
        var a = new Int32ActivationBlock(values, scale);
        var b = new Int32ActivationBlock(new int[1, 4] { { -4, 2, 7, 1 } }, new[] { 0.25f });

        var adder = new IntegerResidualAdder();
        var sum = adder.Add(a, b);

        Assert.Equal(0.25f, sum.RowScales[0]);
        Assert.Equal(6, sum.Values[0, 0]);
        Assert.Equal(-3, sum.Values[0, 1]);
        Assert.Equal(10, sum.Values[0, 2]);
        Assert.Equal(1, sum.Values[0, 3]);
    }

    [Fact]
    public void Add_ZeroOperand_PreservesNonZero()
    {
        var a = new Int32ActivationBlock(
            new int[1, 3] { { 5, -7, 0 } },
            new[] { 0.5f });
        var b = new Int32ActivationBlock(
            new int[1, 3] { { 0, 0, 0 } },
            new[] { 0.001f });

        var sum = new IntegerResidualAdder().Add(a, b);
        var actualFloat = sum.ToFloat();

        Assert.InRange(actualFloat[0, 0] - 2.5f, -1e-4f, 1e-4f);
        Assert.InRange(actualFloat[0, 1] - -3.5f, -1e-4f, 1e-4f);
        Assert.InRange(actualFloat[0, 2], -1e-4f, 1e-4f);
    }

    [Fact]
    public void Add_RejectsShapeMismatch()
    {
        var a = new Int32ActivationBlock(new int[1, 3], new float[] { 1f });
        var b = new Int32ActivationBlock(new int[1, 4], new float[] { 1f });

        Assert.Throws<ArgumentException>(() => new IntegerResidualAdder().Add(a, b));
    }

    [Fact]
    public void Add_RejectsRowCountMismatch()
    {
        var a = new Int32ActivationBlock(new int[2, 3], new float[] { 1f, 1f });
        var b = new Int32ActivationBlock(new int[3, 3], new float[] { 1f, 1f, 1f });

        Assert.Throws<ArgumentException>(() => new IntegerResidualAdder().Add(a, b));
    }

    private static Int32ActivationBlock MakeBlock(int rows, int cols, Random rng, float scaleBase)
    {
        var values = new int[rows, cols];
        var scales = new float[rows];
        for (var r = 0; r < rows; r++)
        {
            scales[r] = scaleBase * (r + 1);
            for (var c = 0; c < cols; c++)
            {
                values[r, c] = rng.Next(-200, 201);
            }
        }
        return new Int32ActivationBlock(values, scales);
    }
}
