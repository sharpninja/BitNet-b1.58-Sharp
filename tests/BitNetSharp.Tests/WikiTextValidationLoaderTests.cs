using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class WikiTextValidationLoaderTests
{
    [Fact]
    public void LoadValidationTokens_returns_nonempty_int_array()
    {
        if (!WikiTextValidationLoader.TryResolveDefaultPath(out var path))
        {
            // Skip deterministically — data file is optional in CI minimal checkouts.
            return;
        }

        var tokens = WikiTextValidationLoader.LoadValidationTokens(path);

        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);
        Assert.All(tokens, id => Assert.True(id >= 0));
    }

    [Fact]
    public void ChunkIntoSequences_respects_seqLen()
    {
        var tokens = Enumerable.Range(0, 17).ToArray();

        var chunks = WikiTextValidationLoader.ChunkIntoSequences(tokens, seqLen: 5).ToArray();

        // 17 / 5 = 3 full chunks (remainder of 2 tokens is dropped — too short to predict).
        Assert.Equal(3, chunks.Length);
        Assert.All(chunks, chunk => Assert.Equal(5, chunk.Length));
        Assert.Equal([0, 1, 2, 3, 4], chunks[0]);
        Assert.Equal([5, 6, 7, 8, 9], chunks[1]);
        Assert.Equal([10, 11, 12, 13, 14], chunks[2]);
    }

    [Fact]
    public void ChunkIntoSequences_throws_on_nonPositive_seqLen()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => WikiTextValidationLoader.ChunkIntoSequences([1, 2, 3], seqLen: 0).ToArray());
    }

    [Fact]
    public void ChunkIntoSequences_yields_nothing_for_tooShort_input()
    {
        var chunks = WikiTextValidationLoader.ChunkIntoSequences([1, 2], seqLen: 5).ToArray();

        Assert.Empty(chunks);
    }
}
