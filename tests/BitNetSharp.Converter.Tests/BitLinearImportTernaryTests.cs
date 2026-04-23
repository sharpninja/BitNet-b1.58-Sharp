using System;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Converter.Tests;

public sealed class BitLinearImportTernaryTests
{
    [Fact]
    public void ImportTernary_StoresGammaAndReconstructsTernaryCollapsedMatrix()
    {
        const int outDim = 2;
        const int inDim = 5;
        var layer = new BitLinear(new BitLinearConfig(inputDimension: inDim, outputDimension: outDim));

        // Row 0: [-1, 0, +1, +1, 0], Row 1: [0, -1, -1, +1, 0]
        sbyte[] trits = { -1, 0, 1, 1, 0, 0, -1, -1, 1, 0 };
        const float gamma = 0.75f;

        layer.ImportTernary(trits, gamma);

        Assert.Equal(gamma, layer.Gamma, 5);

        var recovered = layer.ToFullPrecision();
        Assert.Equal(outDim, recovered.GetLength(0));
        Assert.Equal(inDim, recovered.GetLength(1));

        for (int row = 0; row < outDim; row++)
        {
            for (int col = 0; col < inDim; col++)
            {
                float expected = trits[row * inDim + col] * gamma;
                Assert.Equal(expected, recovered[row, col], 5);
            }
        }
    }

    [Fact]
    public void ImportTernary_LengthMismatch_Throws()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 4, outputDimension: 3));
        sbyte[] wrongSize = new sbyte[11]; // should be 12

        Assert.Throws<ArgumentException>(() => layer.ImportTernary(wrongSize, 0.1f));
    }

    [Fact]
    public void ImportTernary_OutOfRangeTrit_Throws()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 2, outputDimension: 1));
        sbyte[] withTwo = { 0, 2 };

        Assert.Throws<ArgumentException>(() => layer.ImportTernary(withTwo, 0.1f));
    }

    [Fact]
    public void ImportTernary_NegativeGamma_Throws()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 2, outputDimension: 1));
        sbyte[] trits = { -1, 1 };

        Assert.Throws<ArgumentOutOfRangeException>(() => layer.ImportTernary(trits, -0.1f));
    }

    [Fact]
    public void ImportTernary_ZeroGamma_YieldsZeroMatrixAndDoesNotThrow()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 4, outputDimension: 2));
        sbyte[] trits = { -1, 1, -1, 1, 0, 0, 0, 0 };

        layer.ImportTernary(trits, 0f);

        Assert.Equal(0f, layer.Gamma, 5);
        var recovered = layer.ToFullPrecision();
        foreach (var v in recovered)
        {
            Assert.Equal(0f, v, 5);
        }
    }

    [Fact]
    public void ImportTernary_RoundtripMatchesGetTernaryStats()
    {
        var layer = new BitLinear(new BitLinearConfig(inputDimension: 3, outputDimension: 2));
        // 2 rows of [0, -1, +1] -> 2 neg, 2 zero, 2 pos
        sbyte[] trits = { 0, -1, 1, 0, -1, 1 };

        layer.ImportTernary(trits, 0.5f);

        var stats = layer.GetTernaryStats();
        Assert.Equal(2, stats.NegativeCount);
        Assert.Equal(2, stats.ZeroCount);
        Assert.Equal(2, stats.PositiveCount);
    }

    [Fact]
    public void ImportTernary_AnalyticGammaEqualsMeanAbsW()
    {
        // For a synthetic Q2_0-derived tensor with known block composition,
        // the analytic Gamma formula d * (count_q0 + count_q2 + 2*count_q3)
        // summed over blocks divided by total_weights should equal the
        // tensor-level mean(|w_ternary * Gamma|).
        //
        // This test builds trits + Gamma directly (representing what the
        // converter would pass) and verifies the recovered matrix absmean
        // equals Gamma when every trit is nonzero.
        const int outDim = 4;
        const int inDim = 10;
        var layer = new BitLinear(new BitLinearConfig(inputDimension: inDim, outputDimension: outDim));

        var trits = new sbyte[outDim * inDim];
        // Alternate -1, +1 so every position is nonzero and absmean == gamma.
        for (int i = 0; i < trits.Length; i++)
        {
            trits[i] = (sbyte)((i % 2 == 0) ? -1 : 1);
        }
        const float gamma = 0.4321f;

        layer.ImportTernary(trits, gamma);

        var recovered = layer.ToFullPrecision();
        float absSum = 0f;
        int count = 0;
        foreach (var v in recovered)
        {
            absSum += MathF.Abs(v);
            count++;
        }
        float absMean = absSum / count;
        Assert.Equal(gamma, absMean, 4);
    }
}
