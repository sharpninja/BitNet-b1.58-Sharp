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
        Attention = new MultiHeadAttention(config, random);
        PreFeedForwardNorm = new RmsNorm(config.Dimension, config.RmsNormEpsilon);
        FeedForward = new SwiGLUFeedForward(config, random);
    }

    public BitNetConfig Config { get; }

    public RmsNorm PreAttentionNorm { get; }

    public MultiHeadAttention Attention { get; }

    public RmsNorm PreFeedForwardNorm { get; }

    public SwiGLUFeedForward FeedForward { get; }

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var attentionOutput = Attention.Forward(PreAttentionNorm.Forward(input));
        var residual = TensorMath.Add(input, attentionOutput);
        var feedForwardOutput = FeedForward.Forward(PreFeedForwardNorm.Forward(residual));
        return TensorMath.Add(residual, feedForwardOutput);
    }
}
