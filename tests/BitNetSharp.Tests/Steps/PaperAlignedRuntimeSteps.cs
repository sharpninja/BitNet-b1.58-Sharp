using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TechTalk.SpecFlow;

namespace BitNetSharp.Tests.Steps;

[Binding]
public sealed class PaperAlignedRuntimeSteps
{
    private IHostedAgentModel? _model;
    private IHost? _host;
    private BitNetHostSummary? _hostSummary;
    private ChatResponse? _chatResponse;
    private List<ChatResponseUpdate>? _streamUpdates;
    private int _trainedExampleCount;

    [Given(@"the hosted model named ""(.*)""")]
    public void GivenTheHostedModelNamed(string model)
    {
        _model = HostedAgentModelFactory.Create(model, VerbosityLevel.Normal);
    }

    [When(@"I generate a response for the prompt ""(.*)""")]
    public async Task WhenIGenerateAResponseForThePrompt(string prompt)
    {
        Assert.NotNull(_model);
        BuildHost();
        var chatClient = _host!.Services.GetRequiredService<IChatClient>();
        _chatResponse = await chatClient.GetResponseAsync([new ChatMessage(ChatRole.User, prompt)]);
    }

    [Then("the response text should be non-empty")]
    public void ThenTheResponseTextShouldBeNonEmpty()
    {
        Assert.NotNull(_chatResponse);
        Assert.False(string.IsNullOrWhiteSpace(_chatResponse.Text));
    }

    [Then("the response should identify the selected model")]
    public void ThenTheResponseShouldIdentifyTheSelectedModel()
    {
        Assert.NotNull(_model);
        Assert.NotNull(_chatResponse);
        Assert.Equal(_model.ModelId, _chatResponse.ModelId);
    }

    [When(@"I stream a response for the prompt ""(.*)""")]
    public async Task WhenIStreamAResponseForThePrompt(string prompt)
    {
        Assert.NotNull(_model);
        BuildHost();
        var chatClient = _host!.Services.GetRequiredService<IChatClient>();
        _streamUpdates = [];

        await foreach (var update in chatClient.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, prompt)]))
        {
            _streamUpdates.Add(update);
        }
    }

    [Then("the stream should include at least one update")]
    public void ThenTheStreamShouldIncludeAtLeastOneUpdate()
    {
        Assert.NotNull(_streamUpdates);
        Assert.NotEmpty(_streamUpdates);
    }

    [Then("each stream update should identify the selected model")]
    public void ThenEachStreamUpdateShouldIdentifyTheSelectedModel()
    {
        Assert.NotNull(_model);
        Assert.NotNull(_streamUpdates);
        Assert.All(_streamUpdates, update => Assert.Equal(_model.ModelId, update.ModelId));
    }

    [When("I train the selected model on the default dataset")]
    public void WhenITrainTheSelectedModelOnTheDefaultDataset()
    {
        Assert.NotNull(_model);
        Assert.IsAssignableFrom<ITrainableHostedAgentModel>(_model);
        var trainableModel = (ITrainableHostedAgentModel)_model;
        var examples = BitNetTrainingCorpus.CreateDefaultExamples();
        trainableModel.Train(examples, epochs: 3);
        _trainedExampleCount = examples.Count;
    }

    [When("I build the agent host")]
    public void WhenIBuildTheAgentHost()
    {
        Assert.NotNull(_model);
        BuildHost();
    }

    [Then("the host summary should describe the selected model registration")]
    public void ThenTheHostSummaryShouldDescribeTheSelectedModelRegistration()
    {
        Assert.NotNull(_model);
        Assert.NotNull(_hostSummary);
        Assert.Equal(_model.AgentName, _hostSummary.AgentName);
        Assert.Equal(_model.ModelId, _hostSummary.ModelId);
        Assert.Equal("Microsoft Agent Framework", _hostSummary.HostingFramework);
        Assert.Equal(_model.PrimaryLanguage, _hostSummary.PrimaryLanguage);
    }

    [Then("the training run should complete over the default dataset")]
    public void ThenTheTrainingRunShouldCompleteOverTheDefaultDataset()
    {
        Assert.Equal(BitNetTrainingCorpus.CreateDefaultExamples().Count, _trainedExampleCount);
    }

    [AfterScenario]
    public void AfterScenario()
    {
        _host?.Dispose();
        _model?.Dispose();
    }

    private void BuildHost()
    {
        if (_host is not null)
        {
            return;
        }

        Assert.NotNull(_model);
        _host = BitNetAgentHost.Build(_model);
        _hostSummary = _host.Services.GetRequiredService<BitNetHostSummary>();
    }
}
