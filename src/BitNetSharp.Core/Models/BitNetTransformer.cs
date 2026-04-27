using System.Diagnostics;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Core.Models;

public sealed partial class BitNetTransformer
{
    private readonly float[,] _tokenEmbeddings;
    private readonly ILogger<BitNetTransformer> _logger;

    public BitNetTransformer(BitNetConfig config, ILogger<BitNetTransformer> logger, int seed = 42, IProgress<double>? constructionProgress = null, bool skipRandomInit = false)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(logger);

        Config = config;
        _logger = logger;

        // skipRandomInit path: callers who immediately overwrite every weight
        // (BitNetPaperGguf.Load, BitNetPaperCheckpoint.Load) skip the
        // multi-billion-op random fill entirely. BitLinear ctor still
        // allocates zeroed packed buffers so layout + Forward semantics are
        // unchanged; the difference is purely the absence of the seed pass.
        Random? random = skipRandomInit ? null : new Random(seed);
        _tokenEmbeddings = random is null
            ? new float[config.VocabSize, config.Dimension]
            : ParameterInitializer.CreateMatrix(config.VocabSize, config.Dimension, random);
        // Total construction units: layers + 1 (token embed already done) + 1 (FinalNorm) + 1 (OutputHead).
        // Report progress per layer so callers can show a moving bar during the
        // 95-MB-per-layer random-init pass that dominates load time on phone CPUs.
        var layers = new BitNetLayer[config.LayerCount];
        var totalUnits = config.LayerCount + 2;
        for (var i = 0; i < config.LayerCount; i++)
        {
            layers[i] = new BitNetLayer(config, random);
            constructionProgress?.Report((double)(i + 1) / totalUnits);
        }
        Layers = layers;
        FinalNorm = new RmsNorm(config.Dimension, config.RmsNormEpsilon);
        constructionProgress?.Report((double)(config.LayerCount + 1) / totalUnits);
        OutputHead = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.VocabSize), random);
        constructionProgress?.Report(1.0);
    }

    public BitNetConfig Config { get; }

    public BitNetLayer[] Layers { get; }

    public RmsNorm FinalNorm { get; }

    public BitLinear OutputHead { get; }

    /// <summary>
    /// Enumerates every <see cref="BitLinear"/> projection that contributes trainable
    /// master weights: the attention Q/K/V/O projections and feed-forward gate/up/down
    /// projections inside each <see cref="BitNetLayer"/>, followed by the final
    /// <see cref="OutputHead"/>. The order is stable so optimizer state can be paired
    /// by index.
    /// </summary>
    public IEnumerable<BitLinear> EnumerateBitLinearLayers()
    {
        foreach (var layer in Layers)
        {
            yield return layer.Attention.QueryProjection;
            yield return layer.Attention.KeyProjection;
            yield return layer.Attention.ValueProjection;
            yield return layer.Attention.OutputProjection;
            yield return layer.FeedForward.GateProjection;
            yield return layer.FeedForward.UpProjection;
            yield return layer.FeedForward.DownProjection;
        }

        yield return OutputHead;
    }

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

    /// <summary>
    /// Allocates a KV cache sized for this model's attention topology. Each
    /// layer gets a K and V slab of shape [capacity, kvDim] where kvDim is
    /// <c>KvHeadCount * HeadDimension</c> for GQA and
    /// <c>HeadCount * HeadDimension</c> for plain MHA.
    /// </summary>
    public TransformerCache CreateCache(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);

        var kvDim = Config.UsesGroupedQueryAttention
            ? Config.KvHeadCount * Config.HeadDimension
            : Config.HeadCount * Config.HeadDimension;

        // Section B - KV5b: per Config.KvCacheQuantization, allocate either
        // an fp32 LayerKvCache slab (default) or an int8 QuantizedKvLayerCache
        // slab per layer. Both implement IKvCache.
        var layers = new IKvCache[Layers.Length];
        for (var i = 0; i < Layers.Length; i++)
        {
            layers[i] = Config.KvCacheQuantization switch
            {
                KvCacheQuantization.Int8 => new QuantizedKvLayerCache(capacity, kvDim),
                _ => new LayerKvCache(capacity, kvDim),
            };
        }

        return new TransformerCache(layers, capacity);
    }

    /// <summary>
    /// Cache-aware forward over new tokens only. Embeds just
    /// <paramref name="newTokenIds"/>, processes each layer with its per-layer
    /// cache slot starting at <see cref="TransformerCache.PastLength"/>, then
    /// advances the cache past length. Returns logits with one row per new
    /// token.
    /// </summary>
    public float[,] Forward(IReadOnlyList<int> newTokenIds, TransformerCache cache)
    {
        ArgumentNullException.ThrowIfNull(newTokenIds);
        ArgumentNullException.ThrowIfNull(cache);

        if (newTokenIds.Count == 0)
        {
            throw new ArgumentException("At least one token is required.", nameof(newTokenIds));
        }

        if (cache.Layers.Length != Layers.Length)
        {
            throw new ArgumentException(
                $"Cache layer count {cache.Layers.Length} does not match transformer layer count {Layers.Length}.",
                nameof(cache));
        }

        var positionOffset = cache.PastLength;
        var totalLength = positionOffset + newTokenIds.Count;
        if (totalLength > Config.MaxSequenceLength)
        {
            throw new ArgumentException(
                $"Total length {totalLength} exceeds configured max sequence length {Config.MaxSequenceLength}.",
                nameof(newTokenIds));
        }
        if (totalLength > cache.Capacity)
        {
            throw new ArgumentException(
                $"Total length {totalLength} exceeds cache capacity {cache.Capacity}.",
                nameof(newTokenIds));
        }

        var hidden = Embed(newTokenIds);

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < Layers.Length; i++)
        {
            var layerStart = sw.Elapsed.TotalMilliseconds;
            // KV5b: pass through IKvCache; BitNetLayer.Forward dispatches
            // on concrete type (LayerKvCache vs QuantizedKvLayerCache).
            hidden = Layers[i].Forward(hidden, cache.Layers[i], positionOffset);
            if (_logger.IsEnabled(LogLevel.Trace))
            {
                _logger.LogTrace(
                    "Layer[{Layer}].ForwardCached new_rows={NewRows} past_length={PastLength} ms={LayerMs:F2}",
                    i,
                    newTokenIds.Count,
                    positionOffset,
                    sw.Elapsed.TotalMilliseconds - layerStart);
            }
        }
        sw.Stop();

        cache.PastLength = totalLength;

        var finalHidden = FinalNorm.Forward(hidden);
        return OutputHead.Forward(finalHidden);
    }

    public float[,] Forward(IReadOnlyList<int> tokenIds)
    {
        var sw = Stopwatch.StartNew();
        var preHead = ForwardHiddenStates(tokenIds);
        var preHeadMs = sw.Elapsed.TotalMilliseconds;
        sw.Restart();
        var logits = OutputHead.Forward(preHead);
        sw.Stop();
        _logger.LogDebug(
            "Transformer.Forward seq_len={SeqLen} pre_head_ms={PreHeadMs:F1} output_head_ms={OutputHeadMs:F1}",
            tokenIds.Count,
            preHeadMs,
            sw.Elapsed.TotalMilliseconds);
        return logits;
    }

    public float[,] ForwardHiddenStates(IReadOnlyList<int> tokenIds)
    {
        var hidden = ForwardPreHeadStates(tokenIds);
        return FinalNorm.Forward(hidden);
    }

    internal float[,] ForwardPreHeadStates(IReadOnlyList<int> tokenIds)
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
        CacheTokenIds(tokenIds);

        var sw = Stopwatch.StartNew();
        for (var i = 0; i < Layers.Length; i++)
        {
            var layerStart = sw.Elapsed.TotalMilliseconds;
            hidden = Layers[i].Forward(hidden);
            if (_logger.IsEnabled(LogLevel.Trace))
            {
                _logger.LogTrace(
                    "Layer[{Layer}].Forward seq_len={SeqLen} ms={LayerMs:F2}",
                    i,
                    tokenIds.Count,
                    sw.Elapsed.TotalMilliseconds - layerStart);
            }
        }
        sw.Stop();

        return hidden;
    }

    internal float[,] ExportTokenEmbeddings()
    {
        var embeddings = new float[_tokenEmbeddings.GetLength(0), _tokenEmbeddings.GetLength(1)];
        Array.Copy(_tokenEmbeddings, embeddings, _tokenEmbeddings.Length);
        return embeddings;
    }

    internal void ImportTokenEmbeddings(float[,] tokenEmbeddings)
    {
        ArgumentNullException.ThrowIfNull(tokenEmbeddings);

        if (tokenEmbeddings.GetLength(0) != _tokenEmbeddings.GetLength(0)
            || tokenEmbeddings.GetLength(1) != _tokenEmbeddings.GetLength(1))
        {
            throw new ArgumentException(
                $"Expected token embeddings with shape [{_tokenEmbeddings.GetLength(0)}, {_tokenEmbeddings.GetLength(1)}], but received [{tokenEmbeddings.GetLength(0)}, {tokenEmbeddings.GetLength(1)}].",
                nameof(tokenEmbeddings));
        }

        Array.Copy(tokenEmbeddings, _tokenEmbeddings, _tokenEmbeddings.Length);
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
