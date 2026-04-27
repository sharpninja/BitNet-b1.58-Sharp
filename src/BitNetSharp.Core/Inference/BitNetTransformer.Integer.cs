using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Inference;

/// <summary>
/// Integer-path partial of <see cref="BitNetTransformer"/>: routes every
/// <see cref="BitNetLayer"/> through <see cref="IntegerForwardComposer"/>
/// so the float forward path can be progressively retired. Embedding and
/// output head are deliberately left on the float path for phase F1; they
/// are the cleanest boundaries for follow-up integer swaps.
/// </summary>
public static partial class BitNetTransformerIntegerExtensions
{
    /// <summary>
    /// Embeds <paramref name="tokenIds"/> and runs every layer through
    /// <see cref="IntegerForwardComposer.ForwardFullSeq"/>, returning the
    /// pre-head hidden states. Matches <c>ForwardPreHeadStates</c> within
    /// the integer-precision floor compounded over layer depth.
    /// </summary>
    public static float[,] ForwardPreHeadIntegerStates(
        this BitNetTransformer transformer,
        IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(tokenIds);

        if (tokenIds.Count == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(tokenIds));
        }

        if (tokenIds.Count > transformer.Config.MaxSequenceLength)
        {
            throw new ArgumentException(
                $"Sequence length {tokenIds.Count} exceeds configured max sequence length {transformer.Config.MaxSequenceLength}.",
                nameof(tokenIds));
        }

        float[,] hidden = EmbedTokens(transformer, tokenIds);
        for (int i = 0; i < transformer.Layers.Length; i++)
        {
            hidden = IntegerForwardComposer.ForwardFullSeq(transformer.Layers[i], hidden);
        }
        return hidden;
    }

    /// <summary>
    /// Full integer forward: embeds tokens, stacks layers via the integer
    /// composer, applies integer FinalNorm, and emits logits via
    /// <see cref="Layers.BitLinear.ForwardInt32"/>. Returns a float[,] so the
    /// public surface matches <c>BitNetTransformer.Forward(tokenIds)</c>.
    /// Argmax on the last row will match the float reference because
    /// softmax is monotonic.
    /// </summary>
    public static float[,] ForwardInteger(
        this BitNetTransformer transformer,
        IReadOnlyList<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(tokenIds);

        float[,] preHead = transformer.ForwardPreHeadIntegerStates(tokenIds);

        // FinalNorm through integer primitive.
        var finalNormScale = transformer.FinalNorm.ExportScale();
        var intFinalNorm = new IntegerRmsNorm(transformer.Config.Dimension, transformer.Config.RmsNormEpsilon);
        intFinalNorm.ImportScale(finalNormScale);
        float[,] normed = intFinalNorm.Forward(preHead);

        // OutputHead through BitLinear.ForwardInt32 then dequantise so
        // callers see the same float[,] logits shape as Forward(tokenIds).
        var quantNormed = QuantizedActivationBlock.FromFloat(normed);
        return transformer.OutputHead.ForwardInt32(quantNormed).ToFloat();
    }

    /// <summary>
    /// Cache-aware integer forward: mirrors
    /// <see cref="BitNetTransformer.Forward(IReadOnlyList{int}, TransformerCache)"/>
    /// but routes every layer through
    /// <see cref="IntegerForwardComposer.ForwardWithCache"/>. Embeds only the
    /// new tokens, advances <see cref="TransformerCache.PastLength"/>, then
    /// runs integer FinalNorm + OutputHead. This is the per-token decode hot
    /// path, so the single-row case is the one that actually ships.
    /// </summary>
    public static float[,] ForwardWithCacheInteger(
        this BitNetTransformer transformer,
        IReadOnlyList<int> newTokenIds,
        TransformerCache cache)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        ArgumentNullException.ThrowIfNull(newTokenIds);
        ArgumentNullException.ThrowIfNull(cache);

        if (newTokenIds.Count == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(newTokenIds));
        }
        if (cache.Layers.Length != transformer.Layers.Length)
        {
            throw new ArgumentException(
                $"Cache layer count {cache.Layers.Length} does not match transformer layer count {transformer.Layers.Length}.",
                nameof(cache));
        }

        int positionOffset = cache.PastLength;
        int totalLength = positionOffset + newTokenIds.Count;
        if (totalLength > transformer.Config.MaxSequenceLength)
        {
            throw new ArgumentException(
                $"Total length {totalLength} exceeds configured max sequence length {transformer.Config.MaxSequenceLength}.",
                nameof(newTokenIds));
        }
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Total length {totalLength} exceeds cache capacity {cache.Capacity}.",
                nameof(newTokenIds));
        }

        float[,] hidden = EmbedTokens(transformer, newTokenIds);
        for (int i = 0; i < transformer.Layers.Length; i++)
        {
            hidden = IntegerForwardComposer.ForwardWithCache(
                transformer.Layers[i],
                hidden,
                cache.Layers[i],
                positionOffset);
        }

        cache.PastLength = totalLength;

        var finalNormScale = transformer.FinalNorm.ExportScale();
        var intFinalNorm = new IntegerRmsNorm(transformer.Config.Dimension, transformer.Config.RmsNormEpsilon);
        intFinalNorm.ImportScale(finalNormScale);
        float[,] normed = intFinalNorm.Forward(hidden);

        var quantNormed = QuantizedActivationBlock.FromFloat(normed);
        return transformer.OutputHead.ForwardInt32(quantNormed).ToFloat();
    }

    private static float[,] EmbedTokens(BitNetTransformer transformer, IReadOnlyList<int> tokenIds)
    {
        int dim = transformer.Config.Dimension;
        var tokenEmbeddings = transformer.ExportTokenEmbeddings();
        var embeddings = new float[tokenIds.Count, dim];
        for (int tokenIndex = 0; tokenIndex < tokenIds.Count; tokenIndex++)
        {
            int tokenId = tokenIds[tokenIndex];
            if (tokenId < 0 || tokenId >= transformer.Config.VocabSize)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tokenIds),
                    $"Token id {tokenId} is outside the configured vocabulary range.");
            }
            for (int d = 0; d < dim; d++)
            {
                embeddings[tokenIndex, d] = tokenEmbeddings[tokenId, d];
            }
        }
        return embeddings;
    }
}
