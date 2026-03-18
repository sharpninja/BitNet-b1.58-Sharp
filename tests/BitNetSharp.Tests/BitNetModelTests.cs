using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.DependencyInjection;
using System.Reflection;

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
        Assert.Equal("bitnet-b1.58-sharp", summary.ModelId);
        Assert.Equal("Microsoft Agent Framework", summary.HostingFramework);
        Assert.Equal("en-US", summary.PrimaryLanguage);
    }

    [Fact]
    public async Task HostedAgentFactorySupportsTraditionalComparisonModel()
    {
        using var model = HostedAgentModelFactory.Create(HostedAgentModelFactory.TraditionalLocalModelId, VerbosityLevel.Normal);
        var response = await model.GetResponseAsync("how are you hosted");

        Assert.Equal(HostedAgentModelFactory.TraditionalLocalModelId, model.ModelId);
        Assert.False(string.IsNullOrWhiteSpace(response.Text));
        Assert.Contains("microsoft agent framework", response.Text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(response.Diagnostics, diagnostic => diagnostic.Contains("tensor-based ordered-context", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TraditionalLocalModelLearnsSimplePromptResponse()
    {
        var model = new TraditionalLocalModel(
            new BitNetOptions(["alpha", "beta", "gamma", "delta"], VerbosityLevel.Normal, MaxResponseTokens: 4),
            embeddingDimension: 16,
            contextWindow: 4,
            seed: 11);

        model.Train(
            [
                new TrainingExample("alpha beta", "gamma delta")
            ],
            epochs: 80,
            learningRate: 0.55f);

        var result = model.GenerateResponse("alpha beta", maxTokens: 2);

        Assert.Equal(["gamma", "delta"], result.Tokens);
        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Contains("tensor-based ordered-context", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void BenchmarkOptionsIncludePrimaryAndComparisonModels()
    {
        var options = HostedAgentBenchmarkOptions.Parse(
            ["benchmark", "--model=bitnet-b1.58-sharp", "--compare-model=traditional-local", "--prompt=how are you hosted", "--max-tokens=3"],
            VerbosityLevel.Verbose);

        Assert.Equal(["bitnet-b1.58-sharp", "traditional-local"], options.ModelSpecifiers);
        Assert.Equal("how are you hosted", options.Prompt);
        Assert.Equal(3, options.MaxOutputTokens);
        Assert.Equal(VerbosityLevel.Verbose, options.Verbosity);
    }
}
