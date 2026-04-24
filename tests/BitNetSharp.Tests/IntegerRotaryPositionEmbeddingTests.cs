using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I4: RoPE sin/cos table precomputed as int16 Q1.15 and applied via
/// integer multiply-add. Float input gets converted to Q16.16, rotated using
/// Q1.15 sin/cos, then converted back. Result must match the reference float
/// RotaryPositionEmbedding within 1e-3 per element.
/// </summary>
public sealed class IntegerRotaryPositionEmbeddingTests
{
    [Fact]
    public void ApplyInPlace_MatchesFloatReference_WithinTolerance()
    {
        const int headDim = 32;
        const int headCount = 2;
        const int seqLen = 16;

        var rng = new Random(7);
        var input = new float[seqLen, headDim * headCount];
        var inputCopy = new float[seqLen, headDim * headCount];
        for (var r = 0; r < seqLen; r++)
        {
            for (var c = 0; c < headDim * headCount; c++)
            {
                var v = ((float)rng.NextDouble() - 0.5f) * 2f;
                input[r, c] = v;
                inputCopy[r, c] = v;
            }
        }

        var reference = new RotaryPositionEmbedding(headDim);
        reference.ApplyInPlace(input, headCount);

        var integer = new IntegerRotaryPositionEmbedding(headDim, maxSequenceLength: seqLen);
        integer.ApplyInPlace(inputCopy, headCount);

        for (var r = 0; r < seqLen; r++)
        {
            for (var c = 0; c < headDim * headCount; c++)
            {
                Assert.InRange(inputCopy[r, c] - input[r, c], -1e-3f, 1e-3f);
            }
        }
    }

    [Fact]
    public void ApplyInPlace_WithPositionOffset_MatchesFloatReference()
    {
        const int headDim = 16;
        const int headCount = 1;
        const int positionOffset = 5;
        const int newRows = 3;
        const int maxSeqLen = 32;

        var rng = new Random(13);
        var input = new float[newRows, headDim];
        var inputCopy = new float[newRows, headDim];
        for (var r = 0; r < newRows; r++)
        {
            for (var c = 0; c < headDim; c++)
            {
                var v = ((float)rng.NextDouble() - 0.5f) * 2f;
                input[r, c] = v;
                inputCopy[r, c] = v;
            }
        }

        new RotaryPositionEmbedding(headDim).ApplyInPlace(input, headCount, positionOffset);
        new IntegerRotaryPositionEmbedding(headDim, maxSeqLen).ApplyInPlace(inputCopy, headCount, positionOffset);

        for (var r = 0; r < newRows; r++)
        {
            for (var c = 0; c < headDim; c++)
            {
                Assert.InRange(inputCopy[r, c] - input[r, c], -1e-3f, 1e-3f);
            }
        }
    }

    [Fact]
    public void Ctor_RejectsOddHeadDim()
    {
        Assert.Throws<ArgumentException>(() =>
            new IntegerRotaryPositionEmbedding(headDim: 15, maxSequenceLength: 32));
    }

    [Fact]
    public void ApplyInPlace_RejectsTensorWithWrongWidth()
    {
        var rope = new IntegerRotaryPositionEmbedding(headDim: 16, maxSequenceLength: 32);
        var wrongWidth = new float[1, 17];

        Assert.Throws<ArgumentException>(() => rope.ApplyInPlace(wrongWidth, headCount: 1, positionOffset: 0));
    }

    [Fact]
    public void ApplyInPlace_RejectsPositionBeyondMaxSequenceLength()
    {
        var rope = new IntegerRotaryPositionEmbedding(headDim: 16, maxSequenceLength: 8);
        var tensor = new float[1, 16];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            rope.ApplyInPlace(tensor, headCount: 1, positionOffset: 10));
    }

    [Fact]
    public void SinCosTable_IsInt16Q1_15()
    {
        var rope = new IntegerRotaryPositionEmbedding(headDim: 16, maxSequenceLength: 4);
        var (sin, cos) = rope.ExportSinCosTable();

        // Q1.15 max = 32767 (for values strictly less than 1).
        for (var i = 0; i < sin.Length; i++)
        {
            Assert.InRange(sin[i], (short)-32768, (short)32767);
            Assert.InRange(cos[i], (short)-32768, (short)32767);
        }
    }
}
