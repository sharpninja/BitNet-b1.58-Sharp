namespace BitNetSharp.Core.Models;

/// <summary>
/// Training-time backward pass for <see cref="BitNetTransformer"/>.
/// </summary>
/// <remarks>
/// Chains the straight-through estimator (STE) gradient from the output logits back through
/// the output head, final RMS norm, every transformer layer in reverse order, and finally
/// accumulates per-token gradients into the token-embedding matrix. BitLinear sub-layers
/// accumulate master-weight gradients internally; embedding gradients live on this class.
///
/// <para>Call <see cref="ZeroTokenEmbeddingGradients"/> (plus per-layer
/// <see cref="Layers.BitLinear.ZeroGradients"/>) between optimizer steps.</para>
/// </remarks>
public sealed partial class BitNetTransformer
{
    private float[,]? _tokenEmbeddingGradients;
    private int[]? _cachedTokenIds;

    /// <summary>
    /// Runs the backward pass starting from gradient of loss with respect to the output
    /// logits. Populates master-weight gradients on every BitLinear layer for which
    /// <see cref="Layers.BitLinear.InitializeMasterWeights"/> has been called, and accumulates
    /// per-token gradients into the token-embedding gradient matrix.
    /// </summary>
    /// <param name="gradientLogits">Gradient with shape <c>[sequenceLength, vocabSize]</c>.</param>
    /// <returns>Gradient with respect to the embedded input at shape <c>[sequenceLength, dimension]</c>.</returns>
    public float[,] Backward(float[,] gradientLogits)
    {
        ArgumentNullException.ThrowIfNull(gradientLogits);

        if (gradientLogits.GetLength(1) != Config.VocabSize)
        {
            throw new ArgumentException(
                $"Expected gradient width {Config.VocabSize} (vocab size), got {gradientLogits.GetLength(1)}.",
                nameof(gradientLogits));
        }

        // OutputHead: [dim -> vocab]. Backward yields grad wrt FinalNorm output.
        var gradHidden = OutputHead.BackwardSTE(gradientLogits);

        // FinalNorm backward.
        gradHidden = FinalNorm.BackwardSTE(gradHidden);

        // Transformer layers in reverse.
        for (var i = Layers.Length - 1; i >= 0; i--)
        {
            gradHidden = Layers[i].BackwardSTE(gradHidden);
        }

        // Token-embedding backward: each row of gradHidden corresponds to the
        // lookup for the token at that input position; scatter-add back.
        AccumulateTokenEmbeddingGradients(gradHidden);

        return gradHidden;
    }

    /// <summary>
    /// Returns a snapshot of accumulated per-token embedding gradients, or <c>null</c> if
    /// no backward pass has run yet.
    /// </summary>
    public float[,]? ExportTokenEmbeddingGradients()
    {
        if (_tokenEmbeddingGradients is null)
        {
            return null;
        }

        var copy = new float[_tokenEmbeddingGradients.GetLength(0), _tokenEmbeddingGradients.GetLength(1)];
        Array.Copy(_tokenEmbeddingGradients, copy, _tokenEmbeddingGradients.Length);
        return copy;
    }

    /// <summary>
    /// Zeros the accumulated token-embedding gradient matrix. Safe to call before it has
    /// ever been allocated (no-op).
    /// </summary>
    public void ZeroTokenEmbeddingGradients()
    {
        if (_tokenEmbeddingGradients is not null)
        {
            Array.Clear(_tokenEmbeddingGradients);
        }
    }

    /// <summary>
    /// Applies an accumulated update to the token-embedding matrix. Typically called by the
    /// optimizer step after <see cref="Backward"/> has populated the embedding gradient.
    /// </summary>
    /// <param name="update">Update tensor with shape <c>[vocabSize, dimension]</c>.</param>
    public void ApplyTokenEmbeddingUpdate(float[,] update)
    {
        ArgumentNullException.ThrowIfNull(update);

        if (update.GetLength(0) != _tokenEmbeddings.GetLength(0)
            || update.GetLength(1) != _tokenEmbeddings.GetLength(1))
        {
            throw new ArgumentException(
                $"Expected update shape [{_tokenEmbeddings.GetLength(0)}, {_tokenEmbeddings.GetLength(1)}], got [{update.GetLength(0)}, {update.GetLength(1)}].",
                nameof(update));
        }

        for (var t = 0; t < _tokenEmbeddings.GetLength(0); t++)
        {
            for (var d = 0; d < _tokenEmbeddings.GetLength(1); d++)
            {
                _tokenEmbeddings[t, d] += update[t, d];
            }
        }
    }

    private void CacheTokenIds(IReadOnlyList<int> tokenIds)
    {
        var copy = new int[tokenIds.Count];
        for (var i = 0; i < tokenIds.Count; i++)
        {
            copy[i] = tokenIds[i];
        }
        _cachedTokenIds = copy;
    }

    private void AccumulateTokenEmbeddingGradients(float[,] gradEmbeddings)
    {
        if (_cachedTokenIds is null)
        {
            // No cached ids means Forward hasn't been called since construction — nothing to scatter.
            return;
        }

        _tokenEmbeddingGradients ??= new float[_tokenEmbeddings.GetLength(0), _tokenEmbeddings.GetLength(1)];

        var seqLen = gradEmbeddings.GetLength(0);
        var dim = gradEmbeddings.GetLength(1);

        if (seqLen != _cachedTokenIds.Length)
        {
            throw new InvalidOperationException(
                $"Cached token-id length {_cachedTokenIds.Length} does not match gradient rows {seqLen}.");
        }

        for (var position = 0; position < seqLen; position++)
        {
            var tokenId = _cachedTokenIds[position];
            for (var d = 0; d < dim; d++)
            {
                _tokenEmbeddingGradients[tokenId, d] += gradEmbeddings[position, d];
            }
        }
    }
}
