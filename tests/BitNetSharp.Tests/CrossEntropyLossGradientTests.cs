using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class CrossEntropyLossGradientTests
{
    [Fact]
    public void Softmax_Minus_OneHot_MatchesFiniteDifferenceGradient()
    {
        // NLL(logits, target) = -log softmax(logits)[target]
        // Analytic gradient: d NLL / d logit_i = softmax(logits)_i - one_hot(target)_i
        var logits = new float[] { -0.8f, 0.2f, 1.3f, -0.1f, 0.6f, 2.0f, 0.4f, -1.1f };
        var target = 3;
        var probs = new float[logits.Length];
        _ = CrossEntropyLoss.FromLogits(logits, target, probs);

        var analytic = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            analytic[i] = probs[i] - (i == target ? 1f : 0f);
        }

        const float eps = 1e-3f;
        for (var i = 0; i < logits.Length; i++)
        {
            var logitsPlus = (float[])logits.Clone();
            var logitsMinus = (float[])logits.Clone();
            logitsPlus[i] += eps;
            logitsMinus[i] -= eps;

            var scratch = new float[logits.Length];
            var lossPlus = CrossEntropyLoss.FromLogits(logitsPlus, target, scratch);
            var lossMinus = CrossEntropyLoss.FromLogits(logitsMinus, target, scratch);

            var numeric = (float)((lossPlus - lossMinus) / (2.0 * eps));

            Assert.InRange(analytic[i] - numeric, -5e-3f, 5e-3f);
        }
    }

    [Fact]
    public void Gradient_SumsToZero()
    {
        // softmax sums to 1; one_hot(target) sums to 1; their difference sums to 0.
        var logits = new float[] { 0.1f, -0.2f, 0.3f, 0.0f };
        var probs = new float[logits.Length];
        _ = CrossEntropyLoss.FromLogits(logits, 1, probs);

        var gradSum = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            gradSum += probs[i] - (i == 1 ? 1f : 0f);
        }

        Assert.InRange(gradSum, -1e-5f, 1e-5f);
    }
}
