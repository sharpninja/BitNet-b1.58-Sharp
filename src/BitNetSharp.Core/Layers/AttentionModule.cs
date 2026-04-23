namespace BitNetSharp.Core.Layers;

/// <summary>
/// Common base for attention layers (MultiHeadAttention, GroupedQueryAttention).
/// Exposes the four BitLinear projections so trainers, audit passes, and model
/// parameter iteration can treat both attention flavors uniformly.
/// </summary>
public abstract class AttentionModule : Module
{
    public abstract BitLinear QueryProjection { get; }

    public abstract BitLinear KeyProjection { get; }

    public abstract BitLinear ValueProjection { get; }

    public abstract BitLinear OutputProjection { get; }

    public abstract float AttentionScale { get; }

    public virtual bool UsesRotaryPositionEmbedding => true;

    public virtual bool AppliesRotaryPositionEmbeddingToQueriesAndKeysOnly => true;

    public virtual bool UsesCausalAttentionMask => true;

    public abstract long EstimateResidentParameterBytes();
}
