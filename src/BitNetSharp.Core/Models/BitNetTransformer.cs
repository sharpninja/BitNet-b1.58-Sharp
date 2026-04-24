using System.Diagnostics;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Core.Models;

public sealed partial class BitNetTransformer
{
    private readonly float[,] _tokenEmbeddings;
    private readonly ILogger<BitNetTransformer> _logger;

    public BitNetTransformer(BitNetConfig config, ILogger<BitNetTransformer> logger, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(logger);

        Config = config;
        _logger = logger;

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
