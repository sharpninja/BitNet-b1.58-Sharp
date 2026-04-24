using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Utils;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

public sealed class TransformerKvCacheTests
{
    private const float Tolerance = 1e-4f;

    private static BitNetConfig SmallConfig() => new(
        vocabSize: 128,
        dimension: 64,
        hiddenDimension: 128,
        layerCount: 2,
        headCount: 4,
        maxSequenceLength: 32,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 2,
        ropeTheta: 10_000f);

    private static BitNetConfig MhaConfig() => new(
        vocabSize: 128,
        dimension: 64,
        hiddenDimension: 128,
        layerCount: 2,
        headCount: 4,
        maxSequenceLength: 32,
        rmsNormEpsilon: 1e-5f,
        kvHeadCount: 4,
        ropeTheta: 10_000f);

    private static float[,] Random(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var buffer = new float[rows, cols];
        for (var r = 0; r < rows; r++)
        {
            for (var c = 0; c < cols; c++)
            {
                buffer[r, c] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
        }

        return buffer;
    }

    private static float MaxAbsDiff(float[,] lhs, float[,] rhs)
    {
        var m = 0f;
        for (var r = 0; r < lhs.GetLength(0); r++)
        {
            for (var c = 0; c < lhs.GetLength(1); c++)
            {
                var d = MathF.Abs(lhs[r, c] - rhs[r, c]);
                if (d > m)
                {
                    m = d;
                }
            }
        }

        return m;
    }

    [Fact]
    public void RopeWithPositionOffset_MatchesFullThenSlice()
    {
        const int headDim = 8;
        const int headCount = 2;
        const int fullSeq = 6;
        const int sliceStart = 4;
        const int sliceLen = fullSeq - sliceStart;

        var rope = new RotaryPositionEmbedding(headDim, theta: 10_000d);

        var full = Random(fullSeq, headCount * headDim, seed: 11);
        var fullCopy = (float[,])full.Clone();
        rope.ApplyInPlace(full, headCount);

        var slice = new float[sliceLen, headCount * headDim];
        for (var r = 0; r < sliceLen; r++)
        {
            for (var c = 0; c < headCount * headDim; c++)
            {
                slice[r, c] = fullCopy[sliceStart + r, c];
            }
        }

        rope.ApplyInPlace(slice, headCount, positionOffset: sliceStart);

        for (var r = 0; r < sliceLen; r++)
        {
            for (var c = 0; c < headCount * headDim; c++)
            {
                Assert.Equal(full[sliceStart + r, c], slice[r, c], 5);
            }
        }
    }

    [Fact]
    public void GqaWithCache_MatchesFullRecompute()
    {
        var config = SmallConfig();
        var gqaFull = new GroupedQueryAttention(config, new Random(7));
        var gqaCached = new GroupedQueryAttention(config, new Random(7));

        const int seqLen = 5;
        var input = Random(seqLen, config.Dimension, seed: 13);

        var expected = gqaFull.Forward(input);

        var kvDim = config.KvHeadCount * config.HeadDimension;
        var cache = new LayerKvCache(capacity: seqLen, kvDim);

        var inputRow0 = new float[1, config.Dimension];
        for (var c = 0; c < config.Dimension; c++)
        {
            inputRow0[0, c] = input[0, c];
        }
        var actualRow0 = gqaCached.Forward(inputRow0, cache, positionOffset: 0);

        var actualRest = new float[seqLen, config.Dimension];
        for (var c = 0; c < config.Dimension; c++)
        {
            actualRest[0, c] = actualRow0[0, c];
        }

        for (var step = 1; step < seqLen; step++)
        {
            var rowInput = new float[1, config.Dimension];
            for (var c = 0; c < config.Dimension; c++)
            {
                rowInput[0, c] = input[step, c];
            }
            var rowOut = gqaCached.Forward(rowInput, cache, positionOffset: step);
            for (var c = 0; c < config.Dimension; c++)
            {
                actualRest[step, c] = rowOut[0, c];
            }
        }

        var lastRowFull = new float[1, config.Dimension];
        var lastRowCached = new float[1, config.Dimension];
        for (var c = 0; c < config.Dimension; c++)
        {
            lastRowFull[0, c] = expected[seqLen - 1, c];
            lastRowCached[0, c] = actualRest[seqLen - 1, c];
        }

        Assert.True(MaxAbsDiff(lastRowFull, lastRowCached) < Tolerance);
    }

    [Fact]
    public void MhaWithCache_MatchesFullRecompute()
    {
        var config = MhaConfig();
        var mhaFull = new MultiHeadAttention(config, new Random(21));
        var mhaCached = new MultiHeadAttention(config, new Random(21));

        const int seqLen = 4;
        var input = Random(seqLen, config.Dimension, seed: 29);

        var expected = mhaFull.Forward(input);

        var kvDim = config.HeadCount * config.HeadDimension;
        var cache = new LayerKvCache(capacity: seqLen, kvDim);

        var actual = new float[seqLen, config.Dimension];
        for (var step = 0; step < seqLen; step++)
        {
            var rowInput = new float[1, config.Dimension];
            for (var c = 0; c < config.Dimension; c++)
            {
                rowInput[0, c] = input[step, c];
            }
            var rowOut = mhaCached.Forward(rowInput, cache, positionOffset: step);
            for (var c = 0; c < config.Dimension; c++)
            {
                actual[step, c] = rowOut[0, c];
            }
        }

        var diff = MaxAbsDiff(expected, actual);
        Assert.True(diff < Tolerance, $"max diff {diff} >= {Tolerance}");
    }

    [Fact]
    public void Transformer_PrefillThenDecodeLoop_MatchesMonolithicForward()
    {
        var config = SmallConfig();
        var logger = NullLogger<BitNetTransformer>.Instance;
        var transformerFull = new BitNetTransformer(config, logger, seed: 91);
        var transformerCached = new BitNetTransformer(config, logger, seed: 91);

        var prompt = new[] { 1, 4, 7, 9, 13 };
        var decodeSteps = new[] { 14, 18, 21 };
        var allTokens = prompt.Concat(decodeSteps).ToArray();

        var expected = transformerFull.Forward(allTokens);

        var cache = transformerCached.CreateCache(config.MaxSequenceLength);
        var prefillLogits = transformerCached.Forward(prompt, cache);

        Assert.Equal(prompt.Length, prefillLogits.GetLength(0));
        for (var step = 0; step < decodeSteps.Length; step++)
        {
            var logitsRow = transformerCached.Forward(new[] { decodeSteps[step] }, cache);
            Assert.Equal(1, logitsRow.GetLength(0));

            var absolute = prompt.Length + step;
            for (var v = 0; v < config.VocabSize; v++)
            {
                var e = expected[absolute, v];
                var a = logitsRow[0, v];
                Assert.Equal(e, a, 2);
            }
        }

        Assert.Equal(allTokens.Length, cache.PastLength);
    }

    [Fact]
    public void Cache_Rollback_RestoresPastLength()
    {
        var config = SmallConfig();
        var transformer = new BitNetTransformer(config, NullLogger<BitNetTransformer>.Instance, seed: 3);
        var cache = transformer.CreateCache(config.MaxSequenceLength);

        transformer.Forward(new[] { 2, 5, 8 }, cache);
        Assert.Equal(3, cache.PastLength);

        transformer.Forward(new[] { 11 }, cache);
        Assert.Equal(4, cache.PastLength);

        cache.RollbackTo(3);
        Assert.Equal(3, cache.PastLength);

        transformer.Forward(new[] { 12 }, cache);
        Assert.Equal(4, cache.PastLength);
    }
}
