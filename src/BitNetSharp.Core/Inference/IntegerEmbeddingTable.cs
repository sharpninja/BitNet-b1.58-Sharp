namespace BitNetSharp.Core.Inference;

/// <summary>
/// Integer-backed token embedding table. Each of VocabSize rows holds
/// Dimension sbyte values and one float scale; reconstruction is
/// row[i] * scale. Memory footprint is 4x smaller than float[,] and the
/// per-row absmax scale keeps reconstruction error at ~max_abs/127 without
/// imposing a global scale.
/// </summary>
public sealed class IntegerEmbeddingTable
{
    private readonly sbyte[,] _rows;
    private readonly float[] _scales;

    public IntegerEmbeddingTable(int vocabSize, int dimension)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);

        VocabSize = vocabSize;
        Dimension = dimension;
        _rows = new sbyte[vocabSize, dimension];
        _scales = new float[vocabSize];
    }

    public int VocabSize { get; }

    public int Dimension { get; }

    public void WriteRow(int tokenId, ReadOnlySpan<float> values)
    {
        ValidateToken(tokenId);
        ValidateDimension(values.Length);

        var maxAbs = 0f;
        for (var i = 0; i < values.Length; i++)
        {
            var abs = MathF.Abs(values[i]);
            if (abs > maxAbs) maxAbs = abs;
        }

        if (maxAbs == 0f)
        {
            _scales[tokenId] = 0f;
            for (var i = 0; i < values.Length; i++)
            {
                _rows[tokenId, i] = 0;
            }
            return;
        }

        var scale = maxAbs / 127f;
        var inverseScale = 1f / scale;
        _scales[tokenId] = scale;
        for (var i = 0; i < values.Length; i++)
        {
            var q = (int)MathF.Round(values[i] * inverseScale);
            _rows[tokenId, i] = (sbyte)Math.Clamp(q, -127, 127);
        }
    }

    public void ReadRow(int tokenId, Span<float> destination)
    {
        ValidateToken(tokenId);
        ValidateDimension(destination.Length);

        var scale = _scales[tokenId];
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = _rows[tokenId, i] * scale;
        }
    }

    public float[,] Lookup(IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        var result = new float[tokenIds.Count, Dimension];
        for (var i = 0; i < tokenIds.Count; i++)
        {
            var tokenId = tokenIds[i];
            ValidateToken(tokenId);
            var scale = _scales[tokenId];
            for (var d = 0; d < Dimension; d++)
            {
                result[i, d] = _rows[tokenId, d] * scale;
            }
        }
        return result;
    }

    public void ImportFromFloat(float[,] source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (source.GetLength(0) != VocabSize || source.GetLength(1) != Dimension)
        {
            throw new ArgumentException(
                $"Expected shape [{VocabSize}, {Dimension}], got [{source.GetLength(0)}, {source.GetLength(1)}].",
                nameof(source));
        }

        var rowBuffer = new float[Dimension];
        for (var v = 0; v < VocabSize; v++)
        {
            for (var d = 0; d < Dimension; d++)
            {
                rowBuffer[d] = source[v, d];
            }
            WriteRow(v, rowBuffer);
        }
    }

    public void Reset()
    {
        Array.Clear(_rows);
        Array.Clear(_scales);
    }

    private void ValidateToken(int tokenId)
    {
        if ((uint)tokenId >= (uint)VocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId,
                $"Token {tokenId} is outside [0, {VocabSize}).");
        }
    }

    private void ValidateDimension(int length)
    {
        if (length != Dimension)
        {
            throw new ArgumentException(
                $"Expected row length {Dimension}, got {length}.", nameof(length));
        }
    }
}
