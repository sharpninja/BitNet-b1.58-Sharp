using System;
using BitNetSharp.Distributed.Contracts;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Tests pinning the contract layer's <see cref="TaskSizingCalculator"/>
/// math. The coordinator and the worker both reference this helper, so
/// any change to the formula here affects both sides of the wire at
/// once — these tests lock the observed numbers so a future refactor
/// cannot silently move the decision boundary for task sizing.
/// </summary>
public sealed class TaskSizingCalculatorTests
{
    [Fact]
    public void Default_target_duration_is_ten_minutes()
    {
        Assert.Equal(TimeSpan.FromMinutes(10), TaskSizingCalculator.DefaultTargetTaskDuration);
    }

    [Fact]
    public void Fallback_tokens_returned_for_non_positive_throughput()
    {
        Assert.Equal(TaskSizingCalculator.FallbackTokensPerTask, TaskSizingCalculator.RecommendedTokensPerTask(0d));
        Assert.Equal(TaskSizingCalculator.FallbackTokensPerTask, TaskSizingCalculator.RecommendedTokensPerTask(-1d));
    }

    [Fact]
    public void Default_efficiency_produces_150_016_tokens_at_1000_tok_per_sec()
    {
        // 1000 tok/s * 0.25 efficiency * 600 s = 150,000 tokens, rounded
        // up to the next 512-token multiple → 150,016.
        var tokens = TaskSizingCalculator.RecommendedTokensPerTask(1_000d);
        Assert.Equal(150_016L, tokens);
    }

    [Fact]
    public void Throughput_scales_linearly_with_tokens_per_second()
    {
        var slow = TaskSizingCalculator.RecommendedTokensPerTask(1_000d);
        var fast = TaskSizingCalculator.RecommendedTokensPerTask(10_000d);
        var ratio = (double)fast / slow;
        Assert.InRange(ratio, 9.5d, 10.5d);
    }

    [Fact]
    public void Efficiency_override_scales_token_budget_proportionally()
    {
        var defaultEfficiency = TaskSizingCalculator.RecommendedTokensPerTask(1_000d);
        var doubled           = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, fullStepEfficiency: 0.50d);
        Assert.True(doubled > defaultEfficiency);
        var ratio = (double)doubled / defaultEfficiency;
        Assert.InRange(ratio, 1.95d, 2.05d);
    }

    [Fact]
    public void Efficiency_below_0_01_is_clamped_to_0_01()
    {
        var clamped = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, fullStepEfficiency: 0d);
        var explicitClamp = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, fullStepEfficiency: 0.01d);
        Assert.Equal(explicitClamp, clamped);
    }

    [Fact]
    public void Efficiency_above_one_is_clamped_to_one()
    {
        var clamped = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, fullStepEfficiency: 2d);
        var explicitClamp = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, fullStepEfficiency: 1d);
        Assert.Equal(explicitClamp, clamped);
    }

    [Fact]
    public void Result_is_always_a_multiple_of_the_granularity()
    {
        for (var tps = 123; tps <= 5_000; tps += 311)
        {
            var tokens = TaskSizingCalculator.RecommendedTokensPerTask(tps);
            Assert.Equal(0L, tokens % TaskSizingCalculator.DefaultGranularityTokens);
        }
    }

    [Fact]
    public void Custom_granularity_is_honored()
    {
        var tokens = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, granularityTokens: 1024L);
        Assert.Equal(0L, tokens % 1024L);
    }

    [Fact]
    public void Non_positive_granularity_throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TaskSizingCalculator.RecommendedTokensPerTask(1_000d, granularityTokens: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TaskSizingCalculator.RecommendedTokensPerTask(1_000d, granularityTokens: -16));
    }

    [Fact]
    public void Non_positive_target_duration_throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TaskSizingCalculator.RecommendedTokensPerTask(1_000d, targetDuration: TimeSpan.Zero));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            TaskSizingCalculator.RecommendedTokensPerTask(1_000d, targetDuration: TimeSpan.FromSeconds(-1)));
    }

    [Fact]
    public void Shorter_target_duration_yields_smaller_tokens()
    {
        var tenMinutes = TaskSizingCalculator.RecommendedTokensPerTask(1_000d);
        var fiveMinutes = TaskSizingCalculator.RecommendedTokensPerTask(1_000d, targetDuration: TimeSpan.FromMinutes(5));
        Assert.True(fiveMinutes < tenMinutes);
    }
}
