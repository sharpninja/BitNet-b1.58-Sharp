using System;
using System.Globalization;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Snapshot of how fast a worker can chew through the representative
/// ternary-matmul inner loop, as measured by the startup BenchmarkDotNet
/// calibration pass. The coordinator uses this number to size the work unit
/// it dispatches to the worker so that each task is ~<see cref="TargetTaskDuration"/>
/// of compute on THAT box, regardless of whether it is a phone-class ARM core
/// or a 64-thread Threadripper.
/// </summary>
internal sealed record CapabilityReport(
    string WorkerName,
    int CpuThreads,
    double TokensPerSecond,
    TimeSpan CalibrationDuration,
    string BenchmarkId,
    DateTimeOffset MeasuredAt)
{
    /// <summary>
    /// Coordinator target for how long a single task should take on the
    /// reporting worker. 10 minutes balances three competing pressures:
    /// network overhead per task (favors large tasks), staleness of async
    /// gradients (favors small tasks), and graceful shutdown latency when a
    /// spot instance gets reclaimed (favors small tasks).
    /// </summary>
    public static readonly TimeSpan TargetTaskDuration = TimeSpan.FromMinutes(10);

    /// <summary>
    /// Computes the number of training tokens the coordinator should pack
    /// into a single task for this worker so execution time lands near
    /// <see cref="TargetTaskDuration"/>. A safety multiplier accounts for the
    /// fact that the startup calibration measures raw forward ternary matmul
    /// throughput but real training steps also include backward pass, loss,
    /// optimizer update, and gradient serialization. Empirically this multiplier
    /// should be refined against D-2 measurements; the 0.25 default assumes
    /// the full training step is ~4x slower than the calibration workload.
    /// </summary>
    public long RecommendedTokensPerTask(double fullStepEfficiency = 0.25d)
    {
        if (TokensPerSecond <= 0)
        {
            // Fallback for the defensive case where calibration returned zero or
            // negative throughput; 4096 tokens is small enough that a busted
            // worker cannot poison the queue.
            return 4096;
        }

        var targetSeconds = TargetTaskDuration.TotalSeconds;
        var effective = TokensPerSecond * Math.Clamp(fullStepEfficiency, 0.01d, 1d);
        var tokens = effective * targetSeconds;

        // Round up to the next power-of-two-friendly multiple of 512 so the
        // training loop can batch tokens into whole sequences without awkward
        // tail fragments.
        const long granularity = 512;
        var rounded = (long)Math.Ceiling(tokens / granularity) * granularity;
        return Math.Max(granularity, rounded);
    }

    /// <summary>
    /// Human-readable one-line summary suitable for worker startup banners
    /// and coordinator dashboards.
    /// </summary>
    public string ToDisplayString()
    {
        var culture = CultureInfo.InvariantCulture;
        return string.Create(culture, $"{WorkerName}: {TokensPerSecond:N0} tok/s on {CpuThreads} threads, calibration {CalibrationDuration.TotalSeconds:F1}s, target task ~{RecommendedTokensPerTask():N0} tok ({TargetTaskDuration.TotalMinutes:F0} min)");
    }
}
