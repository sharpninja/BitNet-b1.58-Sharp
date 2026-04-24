using BitNetSharp.Core.Inference;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase I9: integer argmax / top-k operate directly on int32 logits. Softmax
/// is monotonic so the sampling decisions do not need the probability values,
/// only the raw ints. Keeps sampling fully in the integer path with zero
/// float math at runtime.
/// </summary>
public sealed class IntegerSamplingTests
{
    [Fact]
    public void Argmax_ReturnsIndexOfLargestLogit()
    {
        var logits = new[] { -3, 7, 2, 11, -8, 10 };
        Assert.Equal(3, IntegerSampling.Argmax(logits));
    }

    [Fact]
    public void Argmax_TiesReturnFirstIndex()
    {
        var logits = new[] { 5, 9, 9, 1 };
        Assert.Equal(1, IntegerSampling.Argmax(logits));
    }

    [Fact]
    public void Argmax_SingleElement_ReturnsZero()
    {
        Assert.Equal(0, IntegerSampling.Argmax(new[] { 42 }));
    }

    [Fact]
    public void Argmax_RejectsEmptySpan()
    {
        Assert.Throws<ArgumentException>(() => IntegerSampling.Argmax(ReadOnlySpan<int>.Empty));
    }

    [Fact]
    public void TopK_ReturnsIndicesSortedByLogitDescending()
    {
        var logits = new[] { 1, 8, 3, 9, 2, 7 };
        var topK = IntegerSampling.TopK(logits, 3);

        Assert.Equal(new[] { 3, 1, 5 }, topK);
    }

    [Fact]
    public void TopK_KEqualsLength_ReturnsAllIndicesSorted()
    {
        var logits = new[] { 5, 1, 9, 3 };
        var topK = IntegerSampling.TopK(logits, 4);

        Assert.Equal(new[] { 2, 0, 3, 1 }, topK);
    }

    [Fact]
    public void TopK_KEqualsOne_MatchesArgmax()
    {
        var logits = new[] { 3, 12, 7, 11, -5 };
        var topK = IntegerSampling.TopK(logits, 1);

        Assert.Single(topK);
        Assert.Equal(IntegerSampling.Argmax(logits), topK[0]);
    }

    [Fact]
    public void TopK_RejectsNonPositiveK()
    {
        var logits = new[] { 1, 2, 3 };
        Assert.Throws<ArgumentOutOfRangeException>(() => IntegerSampling.TopK(logits, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => IntegerSampling.TopK(logits, -1));
    }

    [Fact]
    public void TopK_RejectsKLargerThanVocab()
    {
        var logits = new[] { 1, 2 };
        Assert.Throws<ArgumentOutOfRangeException>(() => IntegerSampling.TopK(logits, 3));
    }
}
