using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.DependencyInjection;

namespace BitNetSharp.Tests;

public sealed class BitNetPaperModelTests
{
    [Fact]
    public void GeneratedResponseUsesPaperAlignedTransformerDiagnostics()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);
        var result = model.GenerateResponse("how are you hosted");

        Assert.Contains("Top next-token predictions:", result.ResponseText, StringComparison.Ordinal);
        Assert.NotEmpty(result.Tokens);
        Assert.Contains("decoder-only transformer", result.Diagnostics[1], StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VisualizeSurfaceIncludesAggregatedTernaryWeightDistribution()
    {
        var model = BitNetBootstrap.CreatePaperModel();
        var stats = model.GetTernaryWeightStats();

        Assert.True(stats.NegativeCount > 0);
        Assert.True(stats.PositiveCount > 0);
        Assert.Equal(stats.TotalCount, stats.NegativeCount + stats.ZeroCount + stats.PositiveCount);
    }

    [Fact]
    public void AgentHostBuildsWithMicrosoftAgentFrameworkRegistration()
    {
        var model = BitNetBootstrap.CreatePaperModel();
        using var host = BitNetAgentHost.Build(model);
        var summary = host.Services.GetRequiredService<BitNetHostSummary>();

        Assert.Equal("bitnet-b1.58-sharp", summary.AgentName);
        Assert.Equal("Microsoft Agent Framework", summary.HostingFramework);
        Assert.Equal("en-US", summary.PrimaryLanguage);
    }
}
