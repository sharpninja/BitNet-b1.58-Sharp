using BitNetSharp.Core.Sampling;

namespace BitNetSharp.Tests;

/// <summary>
/// Red/green oracle for <see cref="SamplingUtilities.ApplyRepetitionPenalty"/>.
///
/// This helper is the only thing that stops argmax decoding from latching
/// onto a single high-logit token forever. If positive logits are not scaled
/// down and negative logits not scaled up for tokens already in the context,
/// the autoregressive loop degenerates into "token 40, token 40, token 40..."
/// the way a raw argmax already did. These tests pin that contract.
/// </summary>
public sealed class SamplingUtilitiesTests
{
    [Fact]
    public void PenaltyOfOne_LeavesLogitsUntouched()
    {
        float[] logits = [1.2f, -0.5f, 3.3f, -2.1f];
        ReadOnlySpan<int> context = [0, 1, 2, 3];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 1f);

        Assert.Equal(new[] { 1.2f, -0.5f, 3.3f, -2.1f }, logits);
    }

    [Fact]
    public void EmptyContext_LeavesLogitsUntouched()
    {
        float[] logits = [1.2f, -0.5f, 3.3f];

        SamplingUtilities.ApplyRepetitionPenalty(logits, ReadOnlySpan<int>.Empty, penalty: 1.5f);

        Assert.Equal(new[] { 1.2f, -0.5f, 3.3f }, logits);
    }

    [Fact]
    public void PositiveLogit_DividedByPenalty()
    {
        float[] logits = [2.0f, 0.0f, 0.0f];
        ReadOnlySpan<int> context = [0];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 2f);

        Assert.Equal(1.0f, logits[0]);
        Assert.Equal(0.0f, logits[1]);
        Assert.Equal(0.0f, logits[2]);
    }

    [Fact]
    public void NegativeLogit_MultipliedByPenalty()
    {
        float[] logits = [-1.5f, 0.0f];
        ReadOnlySpan<int> context = [0];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 2f);

        Assert.Equal(-3.0f, logits[0]);
        Assert.Equal(0.0f, logits[1]);
    }

    [Fact]
    public void ZeroLogit_Unchanged()
    {
        float[] logits = [0.0f, 0.0f];
        ReadOnlySpan<int> context = [0];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 3f);

        Assert.Equal(0.0f, logits[0]);
    }

    [Fact]
    public void TokensNotInContext_Unchanged()
    {
        float[] logits = [5.0f, 5.0f, 5.0f];
        ReadOnlySpan<int> context = [1];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 2f);

        Assert.Equal(5.0f, logits[0]);
        Assert.Equal(2.5f, logits[1]);
        Assert.Equal(5.0f, logits[2]);
    }

    [Fact]
    public void OutOfRangeContextIndices_Ignored()
    {
        float[] logits = [1.0f, 2.0f];
        ReadOnlySpan<int> context = [-1, 5, 0];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 2f);

        Assert.Equal(0.5f, logits[0]);
        Assert.Equal(2.0f, logits[1]);
    }

    [Fact]
    public void RepeatedTokenInContext_PenalizedEachOccurrence()
    {
        float[] logits = [4.0f, 0.0f];
        ReadOnlySpan<int> context = [0, 0];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 2f);

        Assert.Equal(1.0f, logits[0]);
    }

    [Fact]
    public void ArgmaxFlipsWhenIncumbentTokenIsPenalized()
    {
        float[] logits = [3.0f, 2.5f, 1.0f];
        ReadOnlySpan<int> context = [0, 0];

        SamplingUtilities.ApplyRepetitionPenalty(logits, context, penalty: 1.5f);

        var argmax = 0;
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > logits[argmax]) argmax = i;
        }

        Assert.Equal(1, argmax);
    }

    [Fact]
    public void NegativePenalty_Throws()
    {
        float[] logits = [1.0f];
        ReadOnlySpan<int> context = [0];
        var span = logits.AsSpan();
        var ctx = context.ToArray();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            SamplingUtilities.ApplyRepetitionPenalty(logits.AsSpan(), ctx, penalty: -1f);
        });
    }

    [Fact]
    public void ZeroPenalty_Throws()
    {
        float[] logits = [1.0f];
        var ctx = new[] { 0 };

        Assert.Throws<ArgumentOutOfRangeException>(() =>
        {
            SamplingUtilities.ApplyRepetitionPenalty(logits.AsSpan(), ctx, penalty: 0f);
        });
    }
}
