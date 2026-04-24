using BitNetSharp.Core.Inference;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I1: the KV cache backing store switches from float[,] to a paged
/// integer representation. Body rows carry int8 values with an int16 Q-scale
/// per row (coarse, fits the bulk of attention history). Tail rows carry
/// int32 values with an int16 Q-scale per row (fine precision for the
/// most recent tokens, which dominate score softmax mass).
///
/// These tests pin the round-trip and math invariants that later I-stages
/// depend on.
/// </summary>
public sealed class IntegerKvCacheLayerTests
{
    private const int Capacity = 32;
    private const int KvDimension = 16;
    private const int TailRows = 4;

    [Fact]
    public void WriteRow_ReadK_RoundTripsWithinInt8Tolerance()
    {
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);
        var rng = new Random(7);
        var source = CreateRandomRow(rng, KvDimension, scale: 1.5f);

        layer.WriteKRow(row: 10, source);

        var recovered = new float[KvDimension];
        layer.ReadKRow(row: 10, recovered);

        var tolerance = MaxAbs(source) / 127f * 1.5f;
        for (var i = 0; i < KvDimension; i++)
        {
            Assert.InRange(recovered[i] - source[i], -tolerance, tolerance);
        }
    }

    [Fact]
    public void WriteRow_ReadV_RoundTripsWithinInt8Tolerance()
    {
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);
        var rng = new Random(11);
        var source = CreateRandomRow(rng, KvDimension, scale: 0.75f);

        layer.WriteVRow(row: 5, source);

        var recovered = new float[KvDimension];
        layer.ReadVRow(row: 5, recovered);

        var tolerance = MaxAbs(source) / 127f * 1.5f;
        for (var i = 0; i < KvDimension; i++)
        {
            Assert.InRange(recovered[i] - source[i], -tolerance, tolerance);
        }
    }

    [Fact]
    public void TailRows_ProduceTighterReconstructionThanBody()
    {
        // Tail is int32-backed so the quant error is epsilon-class;
        // body is int8 so the quant error is O(max_abs/127).
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);
        var rng = new Random(23);
        var source = CreateRandomRow(rng, KvDimension, scale: 1f);

        // Body row
        layer.WriteKRow(row: 0, source);

        // Tail row (last rows before capacity are tail slots)
        layer.WriteKRow(row: Capacity - 1, source);

        var bodyRecovered = new float[KvDimension];
        var tailRecovered = new float[KvDimension];
        layer.ReadKRow(0, bodyRecovered);
        layer.ReadKRow(Capacity - 1, tailRecovered);

        var bodyError = 0f;
        var tailError = 0f;
        for (var i = 0; i < KvDimension; i++)
        {
            bodyError += MathF.Abs(bodyRecovered[i] - source[i]);
            tailError += MathF.Abs(tailRecovered[i] - source[i]);
        }

        Assert.True(tailError <= bodyError,
            $"Expected tail (int32) error <= body (int8) error; body={bodyError}, tail={tailError}.");
    }

    [Fact]
    public void Reset_ClearsAllRowsAndScales()
    {
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);
        var rng = new Random(31);
        var source = CreateRandomRow(rng, KvDimension, scale: 1f);

        layer.WriteKRow(row: 3, source);
        layer.WriteVRow(row: 3, source);

        layer.Reset();

        var recovered = new float[KvDimension];
        layer.ReadKRow(row: 3, recovered);
        Assert.All(recovered, v => Assert.Equal(0f, v));
        layer.ReadVRow(row: 3, recovered);
        Assert.All(recovered, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void WriteRow_OutOfRange_Throws()
    {
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);
        var row = new float[KvDimension];

        Assert.Throws<ArgumentOutOfRangeException>(() => layer.WriteKRow(Capacity, row));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.WriteVRow(-1, row));
        Assert.Throws<ArgumentOutOfRangeException>(() => layer.ReadKRow(Capacity, row));
    }

    [Fact]
    public void ReadRow_WrongLength_Throws()
    {
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);

        Assert.Throws<ArgumentException>(() => layer.ReadKRow(0, new float[KvDimension - 1]));
        Assert.Throws<ArgumentException>(() => layer.WriteKRow(0, new float[KvDimension + 1]));
    }

    [Fact]
    public void Ctor_RejectsTailLargerThanCapacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IntegerKvCacheLayer(Capacity, KvDimension, Capacity + 1));
    }

    [Fact]
    public void RowDotAgainstQuery_ApproximatesFloatOracle()
    {
        // Attention inner loop will dispatch against int-backed K rows.
        // The fast path dequantises one row at a time, so the score should
        // match a float-oracle dot product within int8 noise.
        var layer = new IntegerKvCacheLayer(Capacity, KvDimension, TailRows);
        var rng = new Random(43);
        var keyRow = CreateRandomRow(rng, KvDimension, scale: 1f);
        layer.WriteKRow(0, keyRow);

        var query = CreateRandomRow(rng, KvDimension, scale: 1f);

        var oracle = 0f;
        for (var i = 0; i < KvDimension; i++)
        {
            oracle += query[i] * keyRow[i];
        }

        var recovered = new float[KvDimension];
        layer.ReadKRow(0, recovered);
        var observed = 0f;
        for (var i = 0; i < KvDimension; i++)
        {
            observed += query[i] * recovered[i];
        }

        var tolerance = MathF.Abs(oracle) * 0.05f + 1e-3f;
        Assert.InRange(observed - oracle, -tolerance, tolerance);
    }

    private static float[] CreateRandomRow(Random rng, int dim, float scale)
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
