using BitNetSharp.Core.Inference;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I6: Softmax via integer exp LUT. Input is float[,] logits (public
/// surface stays float for drop-in compat); internally we find the row-max,
/// shift so every value is <= 0, look up exp(shifted) in a Q16.16 LUT, sum
/// (int32), then divide in Q16.16. Result rows must sum to 1 and track the
/// float reference within 1e-3 per element.
/// </summary>
public sealed class IntegerSoftmaxTests
{
    [Fact]
    public void ApplyToFloat_MatchesReferenceSoftmax_WithinTolerance()
    {
        const int rows = 4;
        const int cols = 16;
        var rng = new Random(11);
        var logits = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                logits[r, c] = ((float)rng.NextDouble() - 0.5f) * 10f;
            }
        }

        var expected = ReferenceSoftmax(logits);
        var integer = new IntegerSoftmax();
        var actual = integer.ApplyToFloat(logits);

        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                Assert.InRange(actual[r, c] - expected[r, c], -1e-3f, 1e-3f);
            }
        }
    }

    [Fact]
    public void ApplyToFloat_RowSumsEqualOne_WithinTolerance()
    {
        var rng = new Random(17);
        var logits = new float[3, 8];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                logits[r, c] = ((float)rng.NextDouble() - 0.5f) * 6f;
            }
        }

        var integer = new IntegerSoftmax();
        var probs = integer.ApplyToFloat(logits);

        for (var r = 0; r < 3; r++)
        {
            var sum = 0.0;
            for (var c = 0; c < 8; c++) sum += probs[r, c];
            Assert.InRange(sum - 1.0, -1e-3, 1e-3);
        }
    }

    [Fact]
    public void ApplyToFloat_LargePositiveShift_DoesNotOverflow()
    {
        var logits = new float[1, 4] { { 1000f, 999f, 998f, 997f } };
        var integer = new IntegerSoftmax();
        var probs = integer.ApplyToFloat(logits);

        var sum = 0.0;
        for (var c = 0; c < 4; c++) sum += probs[0, c];
        Assert.InRange(sum - 1.0, -1e-3, 1e-3);
        // Largest logit should have largest probability.
        Assert.True(probs[0, 0] > probs[0, 1]);
        Assert.True(probs[0, 1] > probs[0, 2]);
        Assert.True(probs[0, 2] > probs[0, 3]);
    }

    [Fact]
    public void ApplyToFloat_VeryNegativeTail_ClampsToZero()
    {
        var logits = new float[1, 3] { { 0f, -50f, -100f } };
        var integer = new IntegerSoftmax();
        var probs = integer.ApplyToFloat(logits);

        Assert.InRange(probs[0, 0] - 1f, -1e-3f, 1e-3f);
        Assert.InRange(probs[0, 1], 0f, 1e-3f);
        Assert.InRange(probs[0, 2], 0f, 1e-3f);
    }

    [Fact]
    public void ApplyToFloat_SingleElementRow_ReturnsOne()
    {
        var logits = new float[2, 1] { { 0f }, { 42f } };
        var integer = new IntegerSoftmax();
        var probs = integer.ApplyToFloat(logits);

        Assert.InRange(probs[0, 0] - 1f, -1e-6f, 1e-6f);
        Assert.InRange(probs[1, 0] - 1f, -1e-6f, 1e-6f);
    }

    [Fact]
    public void Ctor_RejectsInvalidLutSize()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IntegerSoftmax(lutEntries: 0, maxShiftMagnitude: 32f));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IntegerSoftmax(lutEntries: 1024, maxShiftMagnitude: 0f));
    }

    private static float[,] ReferenceSoftmax(float[,] logits)
    {
        var rows = logits.GetLength(0);
        var cols = logits.GetLength(1);
        var output = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            var max = float.NegativeInfinity;
            for (var c = 0; c < cols; c++)
            {
                if (logits[r, c] > max) max = logits[r, c];
            }
            var sum = 0.0;
            for (var c = 0; c < cols; c++)
            {
                var e = Math.Exp(logits[r, c] - max);
                output[r, c] = (float)e;
                sum += e;
            }
            for (var c = 0; c < cols; c++)
            {
                output[r, c] = (float)(output[r, c] / sum);
            }
        }
        return output;
    }
}
