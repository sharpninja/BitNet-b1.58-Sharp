using BitNetSharp.Core.Layers;

namespace BitNetSharp.Tests;

public sealed class RmsNormBackwardTests
{
    [Fact]
    public void BackwardSTE_ShapeMatchesInput()
    {
        var norm = new RmsNorm(dimension: 8);
        var input = new float[3, 8];
        var rng = new Random(42);
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        _ = norm.Forward(input);
        var gradOut = new float[3, 8];
        for (var r = 0; r < 3; r++)
        {
            for (var c = 0; c < 8; c++)
            {
                gradOut[r, c] = 1f;
            }
        }

        var gradIn = norm.BackwardSTE(gradOut);
        Assert.Equal(3, gradIn.GetLength(0));
        Assert.Equal(8, gradIn.GetLength(1));
    }

    [Fact]
    public void BackwardSTE_FiniteDifference_MatchesAnalytical()
    {
        var rng = new Random(7);
        const int dim = 6;
        const int rows = 2;

        var norm = new RmsNorm(dim);
        var input = new float[rows, dim];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < dim; c++)
            {
                input[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        var upstream = new float[rows, dim];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < dim; c++)
            {
                upstream[r, c] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        _ = norm.Forward(input);
        var analytical = norm.BackwardSTE(upstream);

        const float eps = 1e-3f;
        var maxDelta = 0f;
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < dim; c++)
            {
                var original = input[r, c];
                input[r, c] = original + eps;
                var plus = norm.Forward(input);
                var lossPlus = Dot(plus, upstream);

                input[r, c] = original - eps;
                var minus = norm.Forward(input);
                var lossMinus = Dot(minus, upstream);

                input[r, c] = original;

                var num = (lossPlus - lossMinus) / (2f * eps);
                var delta = MathF.Abs(num - analytical[r, c]);
                if (delta > maxDelta)
                {
                    maxDelta = delta;
                }
            }
        }

        Assert.True(maxDelta < 1e-2f, $"RmsNorm FD delta too large: {maxDelta}");
    }

    private static float Dot(float[,] a, float[,] b)
    {
        var s = 0f;
        for (var r = 0; r < a.GetLength(0); r++)
        {
            for (var c = 0; c < a.GetLength(1); c++)
            {
                s += a[r, c] * b[r, c];
            }
        }
        return s;
    }
}
