using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Pure function that maps a worker's measured tokens-per-second
/// throughput onto a recommended per-task token budget. The math lives
/// in the Contracts assembly so the worker, the coordinator, and any
/// future observer all agree on exactly how task sizing is derived from
/// a capability report.
///
/// <para>
/// Philosophy: every worker should spend roughly the same wall-clock
/// duration on each task regardless of hardware, so the pool can
/// equalize engagement and minimize staleness under async SGD. The
/// default target is ten minutes because that balances network
/// round-trip overhead (favors larger tasks) against gradient
/// staleness (favors smaller tasks) against orchestrator reclaim
/// latency when a spot instance disappears (favors smaller tasks).
/// </para>
///
/// <para>
/// The <c>fullStepEfficiency</c> factor accounts for the fact that the
/// startup calibration benchmark only measures the forward ternary
/// matmul inner loop. Real training steps also run the backward pass,
/// loss, optimizer update, and gradient serialization, so the true
/// end-to-end step cost is several times the calibration cost. The
/// 0.25 default assumes the whole step is ~4x slower than calibration;
/// operators can override after D-2 empirical measurements.
/// </para>
/// </summary>
public static class TaskSizingCalculator
{
    /// <summary>Default coordinator target for one task's wall-clock.</summary>
    public static readonly TimeSpan DefaultTargetTaskDuration = TimeSpan.FromMinutes(10);

    /// <summary>Rounding granularity so batch sizes line up on sequence boundaries.</summary>
    public const long DefaultGranularityTokens = 512L;

    /// <summary>Safe minimum handed to a worker whose calibration returned zero.</summary>
    public const long FallbackTokensPerTask = 4096L;

    /// <summary>
    /// Assumed ratio of calibration throughput to full-training-step
    /// throughput. Tune after empirical D-2 measurements.
    /// </summary>
    public const double DefaultFullStepEfficiency = 0.25d;

    /// <summary>
    /// Computes the recommended per-task token budget for a worker
    /// given its measured throughput.
    /// </summary>
    /// <param name="tokensPerSecond">Throughput reported by the worker's
    /// startup calibration.</param>
    /// <param name="targetDuration">Target wall-clock per task. Defaults
    /// to <see cref="DefaultTargetTaskDuration"/>.</param>
    /// <param name="fullStepEfficiency">Scaling factor applied to the
    /// raw throughput to account for backward pass, optimizer, and
    /// serialization overhead. Clamped to [0.01, 1.0].</param>
    /// <param name="granularityTokens">Token rounding granularity so the
    /// result lands on a clean batch boundary. Must be positive.</param>
    public static long RecommendedTokensPerTask(
        double tokensPerSecond,
        TimeSpan? targetDuration = null,
        double fullStepEfficiency = DefaultFullStepEfficiency,
        long granularityTokens = DefaultGranularityTokens)
    {
        if (granularityTokens <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(granularityTokens), "Granularity must be positive.");
        }

        if (tokensPerSecond <= 0d)
        {
            return FallbackTokensPerTask;
        }

        var target = targetDuration ?? DefaultTargetTaskDuration;
        if (target <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(targetDuration), "Target duration must be positive.");
        }

        var clampedEfficiency = Math.Clamp(fullStepEfficiency, 0.01d, 1.0d);
        var effectiveTokensPerSecond = tokensPerSecond * clampedEfficiency;
        var rawTokens = effectiveTokensPerSecond * target.TotalSeconds;
        var rounded = (long)Math.Ceiling(rawTokens / granularityTokens) * granularityTokens;
        return Math.Max(granularityTokens, rounded);
    }
}
