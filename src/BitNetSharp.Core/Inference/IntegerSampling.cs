namespace BitNetSharp.Core.Inference;

/// <summary>
/// Pure-integer sampling primitives for logits. Softmax is monotonic, so
/// argmax and top-k operate directly on the int32 logits; no probability
/// computation is required when the caller only needs token identities.
/// </summary>
public static class IntegerSampling
{
    public static int Argmax(ReadOnlySpan<int> logits)
    {
        if (logits.Length == 0)
        {
            throw new ArgumentException("Logits must contain at least one element.", nameof(logits));
        }

        var bestIdx = 0;
        var bestVal = logits[0];
        for (var i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bestVal)
            {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    public static int[] TopK(ReadOnlySpan<int> logits, int k)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(k);
        if (k > logits.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                $"k ({k}) cannot exceed logits length ({logits.Length}).");
        }

        var indices = new int[logits.Length];
        for (var i = 0; i < logits.Length; i++) indices[i] = i;

        var values = logits.ToArray();
        Array.Sort(values, indices, Comparer<int>.Create((a, b) => b.CompareTo(a)));

        var result = new int[k];
        Array.Copy(indices, result, k);
        return result;
    }
}
