namespace BitNetSharp.Core.Models;

public sealed record BitNetConfig
{
    public BitNetConfig(
        int vocabSize = 32_000,
        int dimension = 256,
        int hiddenDimension = 1_024,
        int layerCount = 4,
        int headCount = 8,
        int maxSequenceLength = 256,
        float rmsNormEpsilon = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDimension);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSequenceLength);
        ArgumentOutOfRangeException.ThrowIfNegative(rmsNormEpsilon);

        if (dimension % headCount != 0)
        {
            throw new ArgumentException("The model dimension must be divisible by the head count.", nameof(dimension));
        }

        if ((dimension / headCount) % 2 != 0)
        {
            throw new ArgumentException("The per-head dimension must be even so rotary embeddings can be applied.", nameof(dimension));
        }

        VocabSize = vocabSize;
        Dimension = dimension;
        HiddenDimension = hiddenDimension;
        LayerCount = layerCount;
        HeadCount = headCount;
        MaxSequenceLength = maxSequenceLength;
        RmsNormEpsilon = rmsNormEpsilon;
    }

    public int VocabSize { get; }

    public int Dimension { get; }

    public int HiddenDimension { get; }

    public int LayerCount { get; }

    public int HeadCount { get; }

    public int MaxSequenceLength { get; }

    public float RmsNormEpsilon { get; }

    public int HeadDimension => Dimension / HeadCount;
}
