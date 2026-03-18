using BitNetSharp.Core;

namespace BitNetSharp.App;

public sealed class TraditionalLocalHostedAgentModel : IHostedAgentModel, ITrainableHostedAgentModel
{
    private readonly TraditionalLocalModel _model;

    public TraditionalLocalHostedAgentModel(VerbosityLevel verbosity)
    {
        Verbosity = verbosity;
        _model = TraditionalLocalModel.CreateDefault(verbosity);
    }

    public string AgentName => ModelId;

    public string ModelId => HostedAgentModelFactory.TraditionalLocalModelId;

    public string DisplayName => "Traditional local tensor language model";

    public string PrimaryLanguage => "en-US";

    public VerbosityLevel Verbosity { get; }

    public string SystemPrompt => "Respond in clear American English using the traditional local comparison model.";

    public IReadOnlyList<string> DescribeModel() =>
    [
        DisplayName,
        $"Model ID: {ModelId}",
        $"Embedding dimension: {_model.EmbeddingDimension}",
        $"Context window: {_model.ContextWindow}",
        "Training: tensor-based softmax next-token optimization over the default corpus",
        "Execution: in-process local comparator using System.Numerics.Tensors"
    ];

    public Task<HostedAgentModelResponse> GetResponseAsync(
        string prompt,
        int? maxOutputTokens = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = _model.GenerateResponse(prompt, maxOutputTokens);
        return Task.FromResult(new HostedAgentModelResponse(result.ResponseText, result.Diagnostics));
    }

    public void Train(IEnumerable<TrainingExample> examples, int epochs = 1)
    {
        _model.Train(examples, Math.Max(TraditionalLocalModel.DefaultTrainingEpochs, epochs));
    }

    public void Dispose()
    {
    }
}
