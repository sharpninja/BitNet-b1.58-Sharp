using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Core.Training;

/// <summary>
/// Flat-parameter-vector helpers used by the distributed training
/// protocol. A BitNet transformer's trainable master parameters are
/// serialized as a single contiguous <see cref="float"/>[] so workers
/// can download the current weight snapshot, train locally, and ship
/// back <c>(new_flat - old_flat)</c> as the gradient payload without
/// any per-tensor wire format.
///
/// <para>
/// Canonical order (stable, must be identical on every worker and the
/// coordinator):
/// <list type="number">
///   <item>Token embeddings, row-major shape <c>[vocabSize, dimension]</c>.</item>
///   <item>Each <see cref="BitLinear"/> in
///     <see cref="BitNetTransformer.EnumerateBitLinearLayers"/> order,
///     master weights row-major shape
///     <c>[outputDimension, inputDimension]</c>.</item>
/// </list>
/// </para>
///
/// <para>
/// <see cref="Pack"/> tolerates a transformer whose BitLinear layers
/// have never had <see cref="BitLinear.InitializeMasterWeights"/>
/// called: in that case each tensor is reconstructed from the packed
/// ternary weights multiplied by <see cref="BitLinear.Gamma"/>. This
/// makes the snapshot path cheap for a freshly-constructed model that
/// has not yet been trained.
/// </para>
/// </summary>
public static class FlatParameterPack
{
    /// <summary>
    /// Returns the expected flat-parameter-vector length for the given
    /// <see cref="BitNetConfig"/>. Equal to
    /// <c>vocabSize * dimension</c> (token embeddings) plus the sum of
    /// every BitLinear master-weight tensor size in
    /// <see cref="BitNetTransformer.EnumerateBitLinearLayers"/> order.
    /// </summary>
    public static int ComputeLength(BitNetConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);

        long total = (long)cfg.VocabSize * cfg.Dimension;

        // Per-layer BitLinear contributions must match the projection
        // dimensions declared by MultiHeadAttention + SwiGLUFeedForward.
        // If those layer shapes change, update this calculation too.
        long perLayer =
            4L * cfg.Dimension * cfg.Dimension       // Q, K, V, O attention projections
            + 2L * cfg.Dimension * cfg.HiddenDimension // Gate, Up (dim -> hidden)
            + 1L * cfg.HiddenDimension * cfg.Dimension; // Down (hidden -> dim)

        total += (long)cfg.LayerCount * perLayer;

        // OutputHead is the final BitLinear yielded by EnumerateBitLinearLayers.
        total += (long)cfg.Dimension * cfg.VocabSize;

        if (total > int.MaxValue)
        {
            throw new InvalidOperationException(
                $"Flat parameter vector length {total} exceeds int.MaxValue; reduce model size.");
        }

        return (int)total;
    }

    /// <summary>
    /// Snapshots all trainable master parameters of <paramref name="transformer"/>
    /// into a single contiguous <see cref="float"/>[] in the canonical order
    /// documented on <see cref="FlatParameterPack"/>.
    /// </summary>
    public static float[] Pack(BitNetTransformer transformer)
    {
        ArgumentNullException.ThrowIfNull(transformer);

        var cfg = transformer.Config;
        var flat = new float[ComputeLength(cfg)];
        var offset = 0;

        // Token embeddings [vocab, dim] row-major.
        var tokenEmbeddings = transformer.ExportTokenEmbeddings();
        var rows = tokenEmbeddings.GetLength(0);
        var cols = tokenEmbeddings.GetLength(1);
        Buffer.BlockCopy(tokenEmbeddings, 0, flat, offset * sizeof(float), rows * cols * sizeof(float));
        offset += rows * cols;

        // BitLinear master weights in canonical order.
        foreach (var layer in transformer.EnumerateBitLinearLayers())
        {
            var outDim = layer.Config.OutputDimension;
            var inDim = layer.Config.InputDimension;
            var count = outDim * inDim;

            var master = layer.ExportMasterWeights();
            if (master is not null)
            {
                Buffer.BlockCopy(master, 0, flat, offset * sizeof(float), count * sizeof(float));
            }
            else
            {
                // Never-trained layer: reconstruct the effective master
                // values from the packed ternary weights * gamma. This
                // matches what InitializeMasterWeights would produce.
                var dense = layer.ToFullPrecision();
                Buffer.BlockCopy(dense, 0, flat, offset * sizeof(float), count * sizeof(float));
            }

            offset += count;
        }

        if (offset != flat.Length)
        {
            throw new InvalidOperationException(
                $"Pack offset {offset} does not match computed length {flat.Length}.");
        }

        return flat;
    }

    /// <summary>
    /// Inverse of <see cref="Pack"/>. Overwrites every trainable master
    /// parameter on <paramref name="transformer"/> from <paramref name="flat"/>
    /// and re-quantizes each BitLinear's packed ternary weights so forward
    /// passes pick up the new values.
    /// </summary>
    public static void Unpack(BitNetTransformer transformer, ReadOnlySpan<float> flat)
    {
        ArgumentNullException.ThrowIfNull(transformer);

        var cfg = transformer.Config;
        var expected = ComputeLength(cfg);
        if (flat.Length != expected)
        {
            throw new ArgumentException(
                $"Flat parameter vector length {flat.Length} does not match expected {expected} for this configuration.",
                nameof(flat));
        }

        var offset = 0;

        // Token embeddings.
        var rows = cfg.VocabSize;
        var cols = cfg.Dimension;
        var embeddings = new float[rows, cols];
        var embeddingCount = rows * cols;
        flat.Slice(offset, embeddingCount).CopyTo(ToSpan(embeddings));
        transformer.ImportTokenEmbeddings(embeddings);
        offset += embeddingCount;

        // BitLinear master weights in canonical order.
        foreach (var layer in transformer.EnumerateBitLinearLayers())
        {
            var outDim = layer.Config.OutputDimension;
            var inDim = layer.Config.InputDimension;
            var count = outDim * inDim;

            var buffer = new float[count];
            flat.Slice(offset, count).CopyTo(buffer);
            layer.ImportMasterWeights(buffer);
            layer.SyncTernaryFromMaster();

            offset += count;
        }

        if (offset != flat.Length)
        {
            throw new InvalidOperationException(
                $"Unpack offset {offset} does not match flat length {flat.Length}.");
        }
    }

    private static Span<float> ToSpan(float[,] matrix)
    {
        // Safe because float[,] is stored contiguously in row-major order
        // and the total element count fits in an int (ComputeLength guards this).
        return System.Runtime.InteropServices.MemoryMarshal.CreateSpan(
            ref matrix[0, 0],
            matrix.Length);
    }
}
