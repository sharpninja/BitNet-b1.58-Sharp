namespace BitNetSharp.Core.Training;

public sealed record BitNetDataLoaderOptions
{
    public BitNetDataLoaderOptions(
        int sequenceLength = 256,
        int batchSize = 4,
        double validationFraction = 0.1d,
        double testFraction = 0d,
        bool shuffle = false,
        bool dropLast = true,
        int seed = 42)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sequenceLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegative(validationFraction);
        ArgumentOutOfRangeException.ThrowIfNegative(testFraction);

        if (validationFraction >= 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(validationFraction), "Validation fraction must be less than 1.");
        }

        if (testFraction >= 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(testFraction), "Test fraction must be less than 1.");
        }

        if (validationFraction + testFraction >= 1d)
        {
            throw new ArgumentException("Validation and test fractions must leave room for training data.", nameof(validationFraction));
        }

        SequenceLength = sequenceLength;
        BatchSize = batchSize;
        ValidationFraction = validationFraction;
        TestFraction = testFraction;
        Shuffle = shuffle;
        DropLast = dropLast;
        Seed = seed;
    }

    public int SequenceLength { get; }

    public int BatchSize { get; }

    public double ValidationFraction { get; }

    public double TestFraction { get; }

    public bool Shuffle { get; }

    public bool DropLast { get; }

    public int Seed { get; }

    public int RawSequenceLength => SequenceLength + 1;

    public double TrainingFraction => 1d - ValidationFraction - TestFraction;
}
