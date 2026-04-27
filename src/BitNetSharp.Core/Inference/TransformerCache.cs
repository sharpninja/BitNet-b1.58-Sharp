namespace BitNetSharp.Core.Inference;

/// <summary>
/// Per-layer slab of cached K/V projections. Rows [0, Count) hold the K/V for
/// positions already processed by this layer; the remaining rows up to
/// <see cref="Capacity"/> are pre-allocated scratch space. Cache slots are owned
/// by the <see cref="TransformerCache"/>.
/// </summary>
public sealed class LayerKvCache : IKvCache
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

    public void WriteKRow(int row, ReadOnlySpan<float> kFloat) => WriteRow(K, row, kFloat);

    public void WriteVRow(int row, ReadOnlySpan<float> vFloat) => WriteRow(V, row, vFloat);

    private void WriteRow(float[,] target, int row, ReadOnlySpan<float> src)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Capacity);
        if (src.Length != KvDimension)
        {
            throw new ArgumentException(
                $"Source length {src.Length} != KvDimension {KvDimension}.", nameof(src));
        }

        for (var i = 0; i < KvDimension; i++)
        {
            target[row, i] = src[i];
        }
    }
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
        : this((IKvCache[])layers, capacity)
    {
        ArgumentNullException.ThrowIfNull(layers);
    }

    public TransformerCache(IKvCache[] layers, int capacity)
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

    /// <summary>
    /// Per-layer cache slabs. Each entry implements <see cref="IKvCache"/>;
    /// the concrete type is either <see cref="LayerKvCache"/> (fp32, default)
    /// or <see cref="QuantizedKvLayerCache"/> (int8, opt-in via
    /// <c>BitNetConfig.KvCacheQuantization</c>). Cache-aware Forward
    /// methods type-dispatch on the concrete entry.
    /// </summary>
    public IKvCache[] Layers { get; }

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
