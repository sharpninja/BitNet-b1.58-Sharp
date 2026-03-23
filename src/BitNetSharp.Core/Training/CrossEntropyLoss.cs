namespace BitNetSharp.Core.Training;

public static class CrossEntropyLoss
{
    private const float ProbabilityFloor = 1e-9f;

    public static double FromProbabilities(ReadOnlySpan<float> probabilities, int targetId)
    {
        ValidateTargetIndex(probabilities.Length, targetId);

        return -Math.Log(Math.Max(probabilities[targetId], ProbabilityFloor));
    }

    public static double FromLogits(ReadOnlySpan<float> logits, int targetId, Span<float> probabilities)
    {
        ValidateTargetIndex(logits.Length, targetId);

        if (probabilities.Length != logits.Length)
        {
            throw new ArgumentException("Probability buffer length must match the logits length.", nameof(probabilities));
        }

        var maxLogit = float.NegativeInfinity;
        for (var index = 0; index < logits.Length; index++)
        {
            maxLogit = MathF.Max(maxLogit, logits[index]);
        }

        var partition = 0f;
        for (var index = 0; index < logits.Length; index++)
        {
            var probabilityMass = MathF.Exp(logits[index] - maxLogit);
            probabilities[index] = probabilityMass;
            partition += probabilityMass;
        }

        if (partition <= 0f)
        {
            var uniform = 1f / logits.Length;
            for (var index = 0; index < probabilities.Length; index++)
            {
                probabilities[index] = uniform;
            }

            return -Math.Log(uniform);
        }

        for (var index = 0; index < probabilities.Length; index++)
        {
            probabilities[index] /= partition;
        }

        return FromProbabilities(probabilities, targetId);
    }

    public static float[] Softmax(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0)
        {
            return [];
        }

        var probabilities = new float[logits.Length];
        FromLogits(logits, 0, probabilities);
        return probabilities;
    }

    private static void ValidateTargetIndex(int length, int targetId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetId);
        if (targetId >= length)
        {
            throw new ArgumentOutOfRangeException(nameof(targetId), "Target id must be within the probability range.");
        }
    }
}
