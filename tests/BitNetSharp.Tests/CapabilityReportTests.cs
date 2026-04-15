using System;
using BitNetSharp.Distributed.Worker;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Byrd-process tests locking the capability report / task-sizing math that
/// governs how the coordinator apportions work across heterogeneous workers.
/// These tests are pure calculation — no BenchmarkDotNet, no filesystem —
/// so they run in the fast lane.
/// </summary>
public class CapabilityReportTests
{
    private static CapabilityReport MakeReport(double tokensPerSecond)
    {
        return new CapabilityReport(
            WorkerName: "test-worker",
            CpuThreads: 8,
            TokensPerSecond: tokensPerSecond,
            CalibrationDuration: TimeSpan.FromSeconds(12),
            BenchmarkId: "Int8TernaryMatMul",
            MeasuredAt: DateTimeOffset.UtcNow);
    }

    [Fact]
    public void TargetTaskDuration_is_ten_minutes_by_contract()
    {
        Assert.Equal(TimeSpan.FromMinutes(10), CapabilityReport.TargetTaskDuration);
    }

    [Fact]
    public void RecommendedTokensPerTask_scales_linearly_with_throughput()
    {
        var slow = MakeReport(tokensPerSecond: 1_000d);
        var fast = MakeReport(tokensPerSecond: 10_000d);

        var slowTokens = slow.RecommendedTokensPerTask();
        var fastTokens = fast.RecommendedTokensPerTask();

        // Fast worker should be given roughly 10x the task of the slow worker
        // (allowing for the 512-token rounding granularity at the margins).
        var ratio = (double)fastTokens / slowTokens;
        Assert.InRange(ratio, 9.5d, 10.5d);
    }

    [Fact]
    public void RecommendedTokensPerTask_honors_the_ten_minute_target_at_default_efficiency()
    {
        // A worker that can do 1000 tok/s raw matmul and has 25% end-to-end
        // training-step efficiency should see task sizing of ~150,000 tokens
        // (1000 * 0.25 * 600 = 150,000), rounded up to 150,016 by the 512
        // granularity.
        var report = MakeReport(tokensPerSecond: 1_000d);
        var tokens = report.RecommendedTokensPerTask();

        Assert.Equal(150_016L, tokens);
    }

    [Fact]
    public void RecommendedTokensPerTask_respects_a_caller_supplied_efficiency_override()
    {
        var report = MakeReport(tokensPerSecond: 1_000d);
        // If an operator's measurements show the backward pass is nearly
        // free (rare), they can hand in a higher efficiency and get a
        // correspondingly larger task.
        var tokensDefault = report.RecommendedTokensPerTask(fullStepEfficiency: 0.25d);
        var tokensDouble  = report.RecommendedTokensPerTask(fullStepEfficiency: 0.50d);

        Assert.True(tokensDouble > tokensDefault);
        // Doubling efficiency should roughly double the token count.
        var ratio = (double)tokensDouble / tokensDefault;
        Assert.InRange(ratio, 1.95d, 2.05d);
    }

    [Fact]
    public void RecommendedTokensPerTask_clamps_efficiency_to_sane_range()
    {
        var report = MakeReport(tokensPerSecond: 1_000d);
        // An efficiency of 0 (or negative) would mean "infinite tokens" in
        // the unbounded formula — the clamp must kick in so the coordinator
        // never hands a crazy task size to a slow worker.
        var tokensAtZero = report.RecommendedTokensPerTask(fullStepEfficiency: 0d);
        var tokensAtOne  = report.RecommendedTokensPerTask(fullStepEfficiency: 1d);

        Assert.True(tokensAtZero > 0);
        Assert.True(tokensAtOne >= tokensAtZero);
    }

    [Fact]
    public void RecommendedTokensPerTask_falls_back_to_a_safe_minimum_for_broken_calibrations()
    {
        var broken = MakeReport(tokensPerSecond: 0d);
        var tokens = broken.RecommendedTokensPerTask();

        Assert.Equal(4096L, tokens);
    }

    [Fact]
    public void RecommendedTokensPerTask_rounds_up_to_the_512_token_granularity()
    {
        var report = MakeReport(tokensPerSecond: 1_234d);
        var tokens = report.RecommendedTokensPerTask();
        Assert.Equal(0L, tokens % 512);
    }

    [Fact]
    public void ToDisplayString_includes_the_key_numbers()
    {
        var report = MakeReport(tokensPerSecond: 3_155d) with { WorkerName = "PAYTON-LEGION2", CpuThreads = 16 };
        var display = report.ToDisplayString();

        Assert.Contains("PAYTON-LEGION2", display);
        Assert.Contains("3,155", display);
        Assert.Contains("16 threads", display);
        Assert.Contains("10 min", display);
    }
}
