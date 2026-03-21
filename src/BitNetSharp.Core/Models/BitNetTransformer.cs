using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Models;

public sealed class BitNetTransformer
{
    private readonly float[,] _tokenEmbeddings;

    public BitNetTransformer(BitNetConfig config, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(config);

        Config = config;

        var random = new Random(seed);
        _tokenEmbeddings = ParameterInitializer.CreateMatrix(config.VocabSize, config.Dimension, random);
        Layers = Enumerable.Range(0, config.LayerCount)
            .Select(_ => new BitNetLayer(config, random))
            .ToArray();
        FinalNorm = new RmsNorm(config.Dimension, config.RmsNormEpsilon);
        OutputHead = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.VocabSize), random);
    }

    public BitNetConfig Config { get; }

    public BitNetLayer[] Layers { get; }

    public RmsNorm FinalNorm { get; }

    public BitLinear OutputHead { get; }

    public long EstimateResidentParameterBytes()
    {
        var total = EstimateTokenEmbeddingBytes()
            + FinalNorm.EstimateResidentParameterBytes()
            + OutputHead.EstimateResidentParameterBytes();

        foreach (var layer in Layers)
        {
            total += layer.EstimateResidentParameterBytes();
        }

        return total;
    }

    public long EstimateTokenEmbeddingBytes() => (long)_tokenEmbeddings.Length * sizeof(float);

    public float[,] Forward(IReadOnlyList<int> tokenIds) =>
        OutputHead.Forward(ForwardHiddenStates(tokenIds));

    public float[,] ForwardHiddenStates(IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);

        if (tokenIds.Count == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(tokenIds));
        }

        if (tokenIds.Count > Config.MaxSequenceLength)
        {
            throw new ArgumentException($"Sequence length {tokenIds.Count} exceeds configured max sequence length {Config.MaxSequenceLength}.", nameof(tokenIds));
        }

        var hidden = Embed(tokenIds);

        foreach (var layer in Layers)
        {
            hidden = layer.Forward(hidden);
        }

        return FinalNorm.Forward(hidden);
    }

    private float[,] Embed(IReadOnlyList<int> tokenIds)
    {
        var embeddings = new float[tokenIds.Count, Config.Dimension];

        for (var tokenIndex = 0; tokenIndex < tokenIds.Count; tokenIndex++)
        {
            var tokenId = tokenIds[tokenIndex];
            if (tokenId < 0 || tokenId >= Config.VocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {tokenId} is outside the configured vocabulary range.");
            }

            for (var dimension = 0; dimension < Config.Dimension; dimension++)
            {
                embeddings[tokenIndex, dimension] = _tokenEmbeddings[tokenId, dimension];
            }
        }

        return embeddings;
    }
}
