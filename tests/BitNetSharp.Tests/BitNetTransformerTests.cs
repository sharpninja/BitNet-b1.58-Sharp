using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

public sealed class BitNetTransformerTests
{
    [Fact]
    public void RmsNorm_NormalizesEachTokenToUnitRootMeanSquare()
    {
        var norm = new RmsNorm(dimension: 2, epsilon: 0f);
        var output = norm.Forward(new float[,]
        {
            { 3f, 4f }
        });

        var rms = MathF.Sqrt((output[0, 0] * output[0, 0] + output[0, 1] * output[0, 1]) / 2f);

        Assert.Equal(1f, rms, 5);
    }

    [Fact]
    public void MultiHeadAttention_ProducesFiniteOutputsWithMatchingShape()
    {
        var config = new BitNetConfig(vocabSize: 32, dimension: 8, hiddenDimension: 16, layerCount: 1, headCount: 2, maxSequenceLength: 8);
        var attention = new MultiHeadAttention(config, new Random(1234));
        var input = new float[,]
        {
            { 0.10f, -0.20f, 0.30f, -0.40f, 0.20f, -0.10f, 0.50f, -0.60f },
            { 0.25f, 0.05f, -0.15f, 0.45f, -0.35f, 0.15f, -0.05f, 0.55f },
            { -0.30f, 0.40f, 0.10f, -0.20f, 0.60f, -0.50f, 0.20f, 0.10f }
        };

        var output = attention.Forward(input);

        Assert.Equal(input.GetLength(0), output.GetLength(0));
        Assert.Equal(config.Dimension, output.GetLength(1));

        foreach (var value in output)
        {
            Assert.True(float.IsFinite(value));
        }
    }

    [Fact]
    public void TransformerForward_ReturnsFiniteLogitsForEachInputToken()
    {
        var config = new BitNetConfig(vocabSize: 16, dimension: 8, hiddenDimension: 16, layerCount: 2, headCount: 2, maxSequenceLength: 8);
        var model = new BitNetTransformer(config, seed: 7);

        var logits = model.Forward([1, 3, 5, 7]);

        Assert.Equal(4, logits.GetLength(0));
        Assert.Equal(config.VocabSize, logits.GetLength(1));

        foreach (var value in logits)
        {
            Assert.True(float.IsFinite(value));
        }
    }
}
