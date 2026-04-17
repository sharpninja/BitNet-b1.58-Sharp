namespace BitNetSharp.Core.Models;

/// <summary>
/// Evaluation-time helpers for <see cref="BitNetTransformer"/>: perplexity measurement on
/// pre-tokenized sequences (e.g. the WikiText-2 validation split).
/// </summary>
/// <remarks>
/// Sequence-length policy:
/// <list type="bullet">
///   <item>Sequences with fewer than two tokens are skipped (nothing to predict).</item>
///   <item>Sequences longer than <see cref="BitNetConfig.MaxSequenceLength"/> are evaluated
///         chunk-by-chunk without padding, overlapping by one token between chunks so every
///         target position is scored exactly once.</item>
///   <item>Shorter sequences are evaluated in-place; no padding is applied.</item>
/// </list>
/// </remarks>
public sealed partial class BitNetTransformer
{
    private const double ProbabilityFloor = 1e-9;

    /// <summary>
    /// Computes corpus perplexity over a collection of pre-tokenized sequences. For each
    /// sequence, accumulates negative log-likelihood of predicting each next token conditioned
    /// on prior tokens, then returns <c>exp(totalNll / totalTargetTokens)</c>.
    /// </summary>
    /// <param name="tokenSequences">Non-null sequences of vocabulary-valid token ids.</param>
    /// <returns>The corpus perplexity, or <c>0</c> if no predictable targets are present.</returns>
    public double CalculatePerplexity(IEnumerable<int[]> tokenSequences)
    {
        ArgumentNullException.ThrowIfNull(tokenSequences);

        var totalNll = 0d;
        var totalTokens = 0L;

        foreach (var sequence in tokenSequences)
        {
            if (sequence is null || sequence.Length < 2)
            {
                continue;
            }

            // Validate every token id lies within vocab; Embed() also checks but we want a
            // clearer error path for eval-time data.
            for (var i = 0; i < sequence.Length; i++)
            {
                if (sequence[i] < 0 || sequence[i] >= Config.VocabSize)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenSequences),
                        $"Token id {sequence[i]} at position {i} is outside the configured vocabulary range [0, {Config.VocabSize}).");
                }
            }

            var chunkStart = 0;
            while (chunkStart < sequence.Length - 1)
            {
                var chunkLength = Math.Min(Config.MaxSequenceLength, sequence.Length - chunkStart);
                var chunk = new int[chunkLength];
                Array.Copy(sequence, chunkStart, chunk, 0, chunkLength);

                var logits = Forward(chunk);

                // Row r of logits predicts token at chunkStart + r + 1.
                for (var row = 0; row < chunkLength - 1; row++)
                {
                    var targetTokenId = sequence[chunkStart + row + 1];
                    totalNll += RowNegativeLogLikelihood(logits, row, targetTokenId);
                    totalTokens++;
                }

                if (chunkStart + chunkLength >= sequence.Length)
                {
                    break;
                }

                // Advance so the next chunk's first row scores the token immediately after
                // the last row of this chunk (chunks overlap by one token).
                chunkStart += chunkLength - 1;
            }
        }

        return totalTokens == 0 ? 0d : Math.Exp(totalNll / totalTokens);
    }

    /// <summary>
    /// Computes corpus perplexity directly from a precomputed logits matrix and target token
    /// stream. Row <c>r</c> of <paramref name="logits"/> scores the prediction for
    /// <paramref name="targets"/><c>[r + 1]</c> (teacher-forcing layout). This is primarily a
    /// unit-test hook that exercises the softmax/NLL math without instantiating a model.
    /// </summary>
    public static double PerplexityFromLogits(float[,] logits, IReadOnlyList<int> targets)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(targets);

        var rowCount = logits.GetLength(0);
        if (targets.Count < 2)
        {
            return 0d;
        }

        var predictableRows = Math.Min(rowCount - 1, targets.Count - 1);
        if (predictableRows <= 0)
        {
            return 0d;
        }

        var totalNll = 0d;
        for (var row = 0; row < predictableRows; row++)
        {
            totalNll += RowNegativeLogLikelihood(logits, row, targets[row + 1]);
        }

        return Math.Exp(totalNll / predictableRows);
    }

    private static double RowNegativeLogLikelihood(float[,] logits, int row, int targetId)
    {
        var vocab = logits.GetLength(1);
        if (targetId < 0 || targetId >= vocab)
        {
            throw new ArgumentOutOfRangeException(
                nameof(targetId),
                $"Target id {targetId} is outside the logits vocabulary range [0, {vocab}).");
        }

        var maxLogit = double.NegativeInfinity;
        for (var column = 0; column < vocab; column++)
        {
            var value = logits[row, column];
            if (value > maxLogit)
            {
                maxLogit = value;
            }
        }

        var partition = 0d;
        var targetMass = 0d;
        for (var column = 0; column < vocab; column++)
        {
            var mass = Math.Exp(logits[row, column] - maxLogit);
            partition += mass;
            if (column == targetId)
            {
                targetMass = mass;
            }
        }

        var probability = partition > 0d
            ? Math.Max(targetMass / partition, ProbabilityFloor)
            : ProbabilityFloor;
        return -Math.Log(probability);
    }
}
