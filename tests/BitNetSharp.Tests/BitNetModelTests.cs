using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.DependencyInjection;

namespace BitNetSharp.Tests;

public sealed class BitNetModelTests
{
    [Fact]
    public void TrainingProducesTernaryWeightsAndLossHistory()
    {
        var model = BitNetModel.CreateDefault();
        var report = new BitNetTrainer(model).TrainDefaults();

        Assert.Equal(3, report.Epochs);
        Assert.Equal(3, report.LossHistory.Count);
        Assert.True(report.NegativeWeights > 0);
        Assert.True(report.ZeroWeights > 0);
        Assert.True(report.PositiveWeights > 0);
    }

    [Fact]
    public void GeneratedResponseUsesAmericanEnglishSeedData()
    {
        var (model, _) = BitNetBootstrap.CreateSeededModel(VerbosityLevel.Normal);
        var result = model.GenerateResponse("how are you hosted");

        Assert.Contains("microsoft", result.ResponseText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("agent", result.ResponseText, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(result.Diagnostics);
    }

    [Fact]
    public void GeneratedResponseUsesTernaryPredictionsForUnmemorizedPrompt()
    {
        var model = new BitNetModel(new BitNetOptions(["alpha", "beta", "gamma"], VerbosityLevel.Quiet));
        model.Train([new TrainingExample("alpha", "beta gamma")], epochs: 1);

        var result = model.GenerateResponse("beta");

        Assert.Equal("gamma", result.ResponseText);
        Assert.Empty(result.Diagnostics);
    }

    [Fact]
    public void VisualizationIncludesChartsAndCsv()
    {
        var (_, report) = BitNetBootstrap.CreateSeededModel();
        var visualization = new BitNetVisualizer().CreateReport(report);

        Assert.Contains("Loss by epoch", visualization.LossChart);
        Assert.Contains("Ternary weight distribution", visualization.WeightHistogram);
        Assert.Contains("epoch,loss", visualization.Csv);
    }

    [Fact]
    public void AgentHostBuildsWithMicrosoftAgentFrameworkRegistration()
    {
        var (model, _) = BitNetBootstrap.CreateSeededModel();
        using var host = BitNetAgentHost.Build(model);
        var summary = host.Services.GetRequiredService<BitNetHostSummary>();

        Assert.Equal("bitnet-b1.58-sharp", summary.AgentName);
        Assert.Equal("Microsoft Agent Framework", summary.HostingFramework);
        Assert.Equal("en-US", summary.PrimaryLanguage);
    }
}
