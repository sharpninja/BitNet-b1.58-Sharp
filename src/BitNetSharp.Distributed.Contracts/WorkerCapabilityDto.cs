using System;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Wire-format capability report that a worker sends to the coordinator
/// during registration. This is the hardware-agnostic score the
/// coordinator uses to size task batches so each unit of work lands close
/// to the target ten-minute compute window on that specific worker.
///
/// This DTO intentionally avoids domain types so it can be freely
/// JSON-serialized and sent over HTTP without pulling transitive
/// references into the Contracts assembly.
/// </summary>
/// <param name="TokensPerSecond">Measured throughput of the reference
/// ternary matmul workload in tokens per wall-clock second.</param>
/// <param name="CpuThreads">Number of logical CPU threads the worker is
/// configured to use.</param>
/// <param name="CalibrationDurationMs">Wall-clock time the BenchmarkDotNet
/// startup calibration consumed, in milliseconds. Sanity signal for
/// operators watching worker logs.</param>
/// <param name="BenchmarkId">Name of the benchmark method that produced
/// the measurement (used for forward-compatibility when the reference
/// workload evolves).</param>
/// <param name="MeasuredAt">UTC timestamp when the calibration finished.</param>
public sealed record WorkerCapabilityDto(
    double TokensPerSecond,
    int CpuThreads,
    long CalibrationDurationMs,
    string BenchmarkId,
    DateTimeOffset MeasuredAt);
