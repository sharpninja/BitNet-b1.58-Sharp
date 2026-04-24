using BitNetSharp.Core.Inference;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I2: token embedding table backing store pivots from float[,] to
/// int8[,] values with one float scale per row. The float table is 4x bigger
/// than it needs to be; per-row scales carry enough magnitude range to keep
/// reconstruction error at ~max_abs/127 without any global scale compromise.
/// </summary>
public sealed class IntegerEmbeddingTableTests
{
    [Fact]
    public void WriteRow_ReadRow_RoundTripsWithinInt8Tolerance()
    {
        var table = new IntegerEmbeddingTable(vocabSize: 8, dimension: 16);
        var rng = new Random(9);
        var source = CreateRow(rng, 16, scale: 1.5f);

        table.WriteRow(tokenId: 3, source);

        var recovered = new float[16];
        table.ReadRow(tokenId: 3, recovered);

        var tolerance = MaxAbs(source) / 127f * 1.5f;
        for (var i = 0; i < 16; i++)
        {
            Assert.InRange(recovered[i] - source[i], -tolerance, tolerance);
        }
    }

    [Fact]
    public void Lookup_EmitsDequantisedRowsForEveryInputToken()
    {
        var table = new IntegerEmbeddingTable(vocabSize: 16, dimension: 8);
        var rng = new Random(13);
        var rows = new float[4][];
        var tokenIds = new[] { 0, 5, 2, 7 };

        for (var i = 0; i < 4; i++)
        {
            rows[i] = CreateRow(rng, 8, scale: 1f);
            table.WriteRow(tokenIds[i], rows[i]);
        }

        var lookup = table.Lookup(tokenIds);

        Assert.Equal(4, lookup.GetLength(0));
        Assert.Equal(8, lookup.GetLength(1));
        for (var i = 0; i < 4; i++)
        {
            var tolerance = MaxAbs(rows[i]) / 127f * 1.5f;
            for (var d = 0; d < 8; d++)
            {
                Assert.InRange(lookup[i, d] - rows[i][d], -tolerance, tolerance);
            }
        }
    }

    [Fact]
    public void ImportFromFloat_ReconstructsEveryRowWithinTolerance()
    {
        var rng = new Random(19);
        var source = new float[12, 8];
        for (var v = 0; v < 12; v++)
        {
            for (var d = 0; d < 8; d++)
            {
                source[v, d] = ((float)rng.NextDouble() - 0.5f) * 2.5f;
            }
        }

        var table = new IntegerEmbeddingTable(vocabSize: 12, dimension: 8);
        table.ImportFromFloat(source);

        for (var v = 0; v < 12; v++)
        {
            var recovered = new float[8];
            table.ReadRow(v, recovered);
            var maxAbs = 0f;
            for (var d = 0; d < 8; d++)
            {
                maxAbs = MathF.Max(maxAbs, MathF.Abs(source[v, d]));
            }
            var tolerance = maxAbs / 127f * 1.5f;
            for (var d = 0; d < 8; d++)
            {
                Assert.InRange(recovered[d] - source[v, d], -tolerance, tolerance);
            }
        }
    }

    [Fact]
    public void Reset_ClearsValuesAndScales()
    {
        var table = new IntegerEmbeddingTable(vocabSize: 4, dimension: 8);
        var rng = new Random(23);
        var row = CreateRow(rng, 8, scale: 1f);

        table.WriteRow(1, row);
        table.Reset();

        var recovered = new float[8];
        table.ReadRow(1, recovered);
        Assert.All(recovered, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void OutOfRange_Throws()
    {
        var table = new IntegerEmbeddingTable(vocabSize: 4, dimension: 8);
        var row = new float[8];

        Assert.Throws<ArgumentOutOfRangeException>(() => table.WriteRow(-1, row));
        Assert.Throws<ArgumentOutOfRangeException>(() => table.WriteRow(4, row));
        Assert.Throws<ArgumentOutOfRangeException>(() => table.ReadRow(4, new float[8]));
    }

    [Fact]
    public void WrongDimension_Throws()
    {
        var table = new IntegerEmbeddingTable(vocabSize: 4, dimension: 8);

        Assert.Throws<ArgumentException>(() => table.WriteRow(0, new float[7]));
        Assert.Throws<ArgumentException>(() => table.ReadRow(0, new float[9]));
    }

    private static float[] CreateRow(Random rng, int dim, float scale)
    {
        var data = new float[dim];
        for (var i = 0; i < dim; i++)
        {
            data[i] = ((float)rng.NextDouble() - 0.5f) * 2f * scale;
        }
        return data;
    }

    private static float MaxAbs(float[] values)
    {
        var m = 0f;
        foreach (var v in values)
        {
            var abs = MathF.Abs(v);
            if (abs > m) m = abs;
        }
        return m;
    }
}
