using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.DependencyInjection;
using System.Reflection;

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
    public void QuantizePreservesLargeCountsBeyondIntRange()
    {
        var model = new BitNetModel(new BitNetOptions(["alpha", "beta", "gamma"], VerbosityLevel.Quiet));
        var getId = typeof(BitNetModel).GetMethod("GetId", BindingFlags.Instance | BindingFlags.NonPublic);
        var quantize = typeof(BitNetModel).GetMethod("Quantize", BindingFlags.Instance | BindingFlags.NonPublic);
        var weightsField = typeof(BitNetModel).GetField("_weights", BindingFlags.Instance | BindingFlags.NonPublic);
        var priorsField = typeof(BitNetModel).GetField("_priors", BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(getId);
        Assert.NotNull(quantize);
        Assert.NotNull(weightsField);
        Assert.NotNull(priorsField);

        var alphaId = (int)getId.Invoke(model, ["alpha"])!;
        var betaId = (int)getId.Invoke(model, ["beta"])!;
        var gammaId = (int)getId.Invoke(model, ["gamma"])!;
        var weights = (sbyte[,])weightsField.GetValue(model)!;
        var vocabularySize = weights.GetLength(0);
        var counts = new long[vocabularySize, vocabularySize];
        var priors = new long[vocabularySize];
        var baseCount = (long)int.MaxValue + 100L;

        for (var column = 0; column < vocabularySize; column++)
        {
            counts[alphaId, column] = baseCount;
            priors[column] = baseCount;
        }

        counts[alphaId, betaId] = baseCount + (vocabularySize * 2L);
        priors[betaId] = baseCount + (vocabularySize * 2L);

        quantize.Invoke(model, [counts, priors]);

        weights = (sbyte[,])weightsField.GetValue(model)!;
        var scoredPriors = (float[])priorsField.GetValue(model)!;

        Assert.Equal(1, weights[alphaId, betaId]);
        Assert.Equal(-1, weights[alphaId, gammaId]);
        Assert.Equal(0.35f, scoredPriors[betaId]);
        Assert.Equal(0f, scoredPriors[gammaId]);
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
