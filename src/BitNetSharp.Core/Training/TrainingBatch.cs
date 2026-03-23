namespace BitNetSharp.Core.Training;

public sealed record TrainingBatch
{
    public TrainingBatch(
        BitNetDataSplit split,
        IReadOnlyList<BitNetTokenSequence> sequences,
        int batchIndex,
        int epochIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(sequences);

        if (sequences.Count == 0)
        {
            throw new ArgumentException("A training batch must contain at least one token sequence.", nameof(sequences));
        }

        Split = split;
        Sequences = sequences;
        BatchIndex = batchIndex;
        EpochIndex = epochIndex;
    }

    public BitNetDataSplit Split { get; }

    public IReadOnlyList<BitNetTokenSequence> Sequences { get; }

    public int BatchIndex { get; }

    public int EpochIndex { get; }

    public int SequenceCount => Sequences.Count;

    public int TokenCount => Sequences.Sum(sequence => sequence.Length);
}
