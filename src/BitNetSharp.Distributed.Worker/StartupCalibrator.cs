using System;
using System.Diagnostics;
using System.IO;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.NoEmit;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Runs <see cref="WorkerCapabilityBenchmark"/> exactly once on worker
/// startup and converts the BenchmarkDotNet summary into a
/// <see cref="CapabilityReport"/> the coordinator can use to size tasks.
///
/// The calibration pass uses the in-process, no-emit toolchain so BDN does
/// not try to spawn a child process (which is awkward inside a minimal
/// Docker runtime image). Iteration counts are kept deliberately tight
/// (1 warmup, 3 iterations, single invocation) so workers finish
/// calibrating in well under a minute even on slow hardware; precision is
/// good enough for coordinator planning, which only needs the order of
/// magnitude right.
/// </summary>
internal static class StartupCalibrator
{
    /// <summary>
    /// Executes the startup calibration benchmark and returns a report
    /// describing this worker's measured throughput. Silences BenchmarkDotNet's
    /// console output so worker logs stay readable.
    /// </summary>
    /// <param name="config">The resolved worker configuration; used only to
    /// stamp the worker name and CPU thread count onto the report.</param>
    public static CapabilityReport Run(WorkerConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        var artifactsPath = Path.Combine(Path.GetTempPath(), "bitnet-worker-bdn");
        Directory.CreateDirectory(artifactsPath);

        var job = Job.Dry
            .WithToolchain(InProcessNoEmitToolchain.Instance)
            .WithWarmupCount(1)
            .WithIterationCount(3)
            .WithUnrollFactor(1)
            .WithInvocationCount(1)
            .WithGcServer(true)
            .WithGcConcurrent(true);

        var benchmarkConfig = ManualConfig.CreateEmpty()
            .AddJob(job)
            .AddLogger(NullLogger.Instance)
            .WithOptions(ConfigOptions.DisableLogFile | ConfigOptions.DisableOptimizationsValidator)
            .WithArtifactsPath(artifactsPath);

        var stopwatch = Stopwatch.StartNew();
        var summary = BenchmarkRunner.Run<WorkerCapabilityBenchmark>(benchmarkConfig);
        stopwatch.Stop();

        if (summary is null || summary.HasCriticalValidationErrors || summary.Reports.Length == 0)
        {
            return BuildFallbackReport(config, stopwatch.Elapsed, "summary-unavailable");
        }

        var benchmarkReport = summary.Reports[0];
        if (!benchmarkReport.Success || benchmarkReport.ResultStatistics is null)
        {
            return BuildFallbackReport(config, stopwatch.Elapsed, benchmarkReport.BenchmarkCase.Descriptor.WorkloadMethod.Name);
        }

        // BenchmarkDotNet already divides by OperationsPerInvoke, so the
        // median measurement is nanoseconds per token. Convert to tok/s.
        var nsPerToken = benchmarkReport.ResultStatistics.Median;
        var tokensPerSecond = nsPerToken > 0d ? 1_000_000_000d / nsPerToken : 0d;

        return new CapabilityReport(
            WorkerName: config.WorkerName,
            CpuThreads: config.CpuThreads,
            TokensPerSecond: tokensPerSecond,
            CalibrationDuration: stopwatch.Elapsed,
            BenchmarkId: benchmarkReport.BenchmarkCase.Descriptor.WorkloadMethod.Name,
            MeasuredAt: DateTimeOffset.UtcNow);
    }

    private static CapabilityReport BuildFallbackReport(WorkerConfig config, TimeSpan elapsed, string benchmarkId)
    {
        return new CapabilityReport(
            WorkerName: config.WorkerName,
            CpuThreads: config.CpuThreads,
            TokensPerSecond: 0d,
            CalibrationDuration: elapsed,
            BenchmarkId: benchmarkId,
            MeasuredAt: DateTimeOffset.UtcNow);
    }
}
