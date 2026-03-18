using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TechTalk.SpecFlow;

namespace BitNetSharp.Tests.Steps;

[Binding]
public sealed class PaperAlignedRuntimeSteps
{
    private BitNetPaperModel? _model;
    private BitNetGenerationResult? _generationResult;
    private TernaryWeightStats? _weightStats;
    private IHost? _host;
    private BitNetHostSummary? _hostSummary;

    [Given("the default paper-aligned BitNet model")]
    public void GivenTheDefaultPaperAlignedBitNetModel()
    {
        _model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);
    }

    [When(@"I generate a response for the prompt ""(.*)""")]
    public void WhenIGenerateAResponseForThePrompt(string prompt)
    {
        Assert.NotNull(_model);
        _generationResult = _model.GenerateResponse(prompt);
    }

    [Then("the response should list top next-token predictions")]
    public void ThenTheResponseShouldListTopNextTokenPredictions()
    {
        Assert.NotNull(_generationResult);
        Assert.Contains("Top next-token predictions:", _generationResult.ResponseText, StringComparison.Ordinal);
    }

    [Then("the response should include generated tokens")]
    public void ThenTheResponseShouldIncludeGeneratedTokens()
    {
        Assert.NotNull(_generationResult);
        Assert.NotEmpty(_generationResult.Tokens);
    }

    [Then("the diagnostics should describe the decoder-only transformer")]
    public void ThenTheDiagnosticsShouldDescribeTheDecoderOnlyTransformer()
    {
        Assert.NotNull(_generationResult);
        Assert.Contains(
            _generationResult.Diagnostics,
            diagnostic => diagnostic.Contains("decoder-only transformer", StringComparison.OrdinalIgnoreCase));
    }

    [When("I inspect the ternary weight distribution")]
    public void WhenIInspectTheTernaryWeightDistribution()
    {
        Assert.NotNull(_model);
        _weightStats = _model.GetTernaryWeightStats();
    }

    [Then("the ternary distribution should include negative, zero, and positive counts")]
    public void ThenTheTernaryDistributionShouldIncludeNegativeZeroAndPositiveCounts()
    {
        Assert.NotNull(_weightStats);
        Assert.Equal(
            _weightStats.TotalCount,
            _weightStats.NegativeCount + _weightStats.ZeroCount + _weightStats.PositiveCount);
    }

    [Then("the ternary distribution should include both negative and positive weights")]
    public void ThenTheTernaryDistributionShouldIncludeBothNegativeAndPositiveWeights()
    {
        Assert.NotNull(_weightStats);
        Assert.True(_weightStats.NegativeCount > 0);
        Assert.True(_weightStats.PositiveCount > 0);
    }

    [When("I build the agent host")]
    public void WhenIBuildTheAgentHost()
    {
        Assert.NotNull(_model);
        _host = BitNetAgentHost.Build(_model);
        _hostSummary = _host.Services.GetRequiredService<BitNetHostSummary>();
    }

    [Then("the host summary should describe the BitNet agent registration")]
    public void ThenTheHostSummaryShouldDescribeTheBitNetAgentRegistration()
    {
        Assert.NotNull(_hostSummary);
        Assert.Equal("bitnet-b1.58-sharp", _hostSummary.AgentName);
        Assert.Equal("Microsoft Agent Framework", _hostSummary.HostingFramework);
        Assert.Equal("en-US", _hostSummary.PrimaryLanguage);
    }

    [AfterScenario]
    public void AfterScenario()
    {
        _host?.Dispose();
    }
}
