using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Layers;

public sealed class BitNetLayer : Module
{
    public BitNetLayer(BitNetConfig config, Random random)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);

        Config = config;
        PreAttentionNorm = new RmsNorm(config.Dimension, config.RmsNormEpsilon);
        Attention = config.UsesGroupedQueryAttention
            ? new GroupedQueryAttention(config, random)
            : new MultiHeadAttention(config, random);
        PreFeedForwardNorm = new RmsNorm(config.Dimension, config.RmsNormEpsilon);
        FeedForward = new SwiGLUFeedForward(config, random);
    }

    public BitNetConfig Config { get; }

    public RmsNorm PreAttentionNorm { get; }

    public AttentionModule Attention { get; }

    public RmsNorm PreFeedForwardNorm { get; }

    public SwiGLUFeedForward FeedForward { get; }

    public long EstimateResidentParameterBytes() =>
        PreAttentionNorm.EstimateResidentParameterBytes()
        + Attention.EstimateResidentParameterBytes()
        + PreFeedForwardNorm.EstimateResidentParameterBytes()
        + FeedForward.EstimateResidentParameterBytes();

    // Cached for backward pass
    private float[,]? _cachedInput;
    private float[,]? _cachedResidual1;

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        _cachedInput = (float[,])input.Clone();

        var attentionOutput = Attention.Forward(PreAttentionNorm.Forward(input));
        var residual = TensorMath.Add(input, attentionOutput);
        _cachedResidual1 = (float[,])residual.Clone();

        var feedForwardOutput = FeedForward.Forward(PreFeedForwardNorm.Forward(residual));
        return TensorMath.Add(residual, feedForwardOutput);
    }

    public float[,] Forward(float[,] input, LayerKvCache cache, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);

        var normed = PreAttentionNorm.Forward(input);
        var attentionOutput = input.GetLength(0) == 1
            ? Attention.ForwardFlashDecode(normed, cache, positionOffset)
            : Attention.Forward(normed, cache, positionOffset);
        var residual = TensorMath.Add(input, attentionOutput);
        var feedForwardOutput = FeedForward.Forward(PreFeedForwardNorm.Forward(residual));
        return TensorMath.Add(residual, feedForwardOutput);
    }

    /// <summary>
    /// Section B - KV5b: cache-type-aware overload that dispatches to the
    /// fp32 or int8 attention path based on the concrete <see cref="IKvCache"/>
    /// implementation.
    /// </summary>
    public float[,] Forward(float[,] input, IKvCache cache, int positionOffset)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(cache);

        return cache switch
        {
            LayerKvCache fp32 => Forward(input, fp32, positionOffset),
            QuantizedKvLayerCache int8 => ForwardInt8(input, int8, positionOffset),
            _ => throw new NotSupportedException(
                $"Unsupported IKvCache implementation {cache.GetType().Name}; expected LayerKvCache or QuantizedKvLayerCache."),
        };
    }

    private float[,] ForwardInt8(float[,] input, QuantizedKvLayerCache cache, int positionOffset)
    {
        var normed = PreAttentionNorm.Forward(input);
        var attentionOutput = input.GetLength(0) == 1
            ? Attention.ForwardFlashDecode(normed, cache, positionOffset)
            : Attention.Forward(normed, cache, positionOffset);
        var residual = TensorMath.Add(input, attentionOutput);
        var feedForwardOutput = FeedForward.Forward(PreFeedForwardNorm.Forward(residual));
        return TensorMath.Add(residual, feedForwardOutput);
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);

        // Backward through second residual: grad flows to both FFN path and residual1
        var gradFeedForward = FeedForward.BackwardSTE(
            PreFeedForwardNorm.BackwardSTE(gradientOutput));

        // Gradient through second residual: gradResidual1 = gradOutput + gradFeedForward
        var rows = gradientOutput.GetLength(0);
        var dim = Config.Dimension;
        var gradResidual1 = new float[rows, dim];
        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < dim; col++)
            {
                gradResidual1[row, col] = gradientOutput[row, col] + gradFeedForward[row, col];
            }
        }

        // Backward through attention path
        var gradAttention = Attention.BackwardSTE(
            PreAttentionNorm.BackwardSTE(gradResidual1));

        // Gradient through first residual: gradInput = gradResidual1 + gradAttention
        var gradInput = new float[rows, dim];
        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < dim; col++)
            {
                gradInput[row, col] = gradResidual1[row, col] + gradAttention[row, col];
            }
        }

        return gradInput;
    }
}
