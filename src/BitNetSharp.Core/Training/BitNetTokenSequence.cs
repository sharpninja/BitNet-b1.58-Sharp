namespace BitNetSharp.Core.Training;

public sealed record BitNetTokenSequence
{
    public BitNetTokenSequence(
        BitNetDataSplit split,
        IReadOnlyList<int> tokenIds,
        string? source = null)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        if (tokenIds.Count < 2)
        {
            throw new ArgumentException("Token sequences must contain at least two tokens for next-token training.", nameof(tokenIds));
        }

        Split = split;
        TokenIds = tokenIds;
        Source = source;
    }

    public BitNetDataSplit Split { get; }

    public IReadOnlyList<int> TokenIds { get; }

    public string? Source { get; }

    public int Length => TokenIds.Count;
}
