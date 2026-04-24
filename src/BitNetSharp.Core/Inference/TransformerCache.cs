namespace BitNetSharp.Core.Inference;

/// <summary>
/// Per-layer slab of cached K/V projections. Rows [0, Count) hold the K/V for
/// positions already processed by this layer; the remaining rows up to
/// <see cref="Capacity"/> are pre-allocated scratch space. Cache slots are owned
/// by the <see cref="TransformerCache"/>.
/// </summary>
public sealed class LayerKvCache
{
    public LayerKvCache(int capacity, int kvDimension)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kvDimension);

        Capacity = capacity;
        KvDimension = kvDimension;
        K = new float[capacity, kvDimension];
        V = new float[capacity, kvDimension];
    }

    public int Capacity { get; }

    public int KvDimension { get; }

    public float[,] K { get; }

    public float[,] V { get; }
}

/// <summary>
/// Request-scoped KV cache for a <see cref="Models.BitNetTransformer"/>.
/// The cache spans every layer; <see cref="PastLength"/> is the number of tokens
/// already processed and therefore already written into every layer's K/V
/// buffers. Prefill adds a batch of rows, decode adds one row at a time,
/// chain-bucket rollback simply resets <see cref="PastLength"/> (stale rows are
/// overwritten on the next write).
/// </summary>
public sealed class TransformerCache
{
    public TransformerCache(LayerKvCache[] layers, int capacity)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);

        foreach (var layer in layers)
        {
            if (layer.Capacity != capacity)
            {
                throw new ArgumentException(
                    $"Layer cache capacity {layer.Capacity} does not match transformer cache capacity {capacity}.",
                    nameof(layers));
            }
        }

        Layers = layers;
        Capacity = capacity;
        PastLength = 0;
    }

    public LayerKvCache[] Layers { get; }

    public int Capacity { get; }

    public int PastLength { get; set; }

    public void Reset() => PastLength = 0;

    public void RollbackTo(int pastLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(pastLength);
        if (pastLength > PastLength)
        {
            throw new ArgumentOutOfRangeException(nameof(pastLength), "Cannot roll forward via RollbackTo.");
        }

        PastLength = pastLength;
    }
}
