using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.App;

public sealed class TraditionalLocalHostedAgentModel : IHostedAgentModel, IInspectableHostedAgentModel, ITrainableHostedAgentModel
{
    private readonly string _trainingCorpusDescription;

    public TraditionalLocalHostedAgentModel(VerbosityLevel verbosity, IEnumerable<TrainingExample>? trainingExamples = null)
    {
        Verbosity = verbosity;
        _trainingCorpusDescription = trainingExamples is null
            ? "default corpus"
            : BitNetTrainingCorpus.BenchmarkDatasetName;
        Model = trainingExamples is null
            ? TraditionalLocalModel.CreateDefault(verbosity)
            : TraditionalLocalModel.CreateForTrainingCorpus(trainingExamples, verbosity);
    }

    public TraditionalLocalModel Model { get; }

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
        $"Embedding dimension: {Model.EmbeddingDimension}",
        $"Context window: {Model.ContextWindow}",
        $"Training: tensor-based softmax next-token optimization over the {_trainingCorpusDescription}",
        "Execution: in-process local comparator using System.Numerics.Tensors"
    ];

    public Task<HostedAgentModelResponse> GetResponseAsync(
        string prompt,
        int? maxOutputTokens = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = Model.GenerateResponse(prompt, maxOutputTokens);
        return Task.FromResult(new HostedAgentModelResponse(result.ResponseText, result.Diagnostics));
    }

    public TernaryWeightStats GetTernaryWeightStats() => Model.GetTernaryWeightStats();

    public void Train(IEnumerable<TrainingExample> examples, int epochs = 1)
    {
        Model.Train(examples, Math.Max(TraditionalLocalModel.DefaultTrainingEpochs, epochs));
    }

    public void Dispose()
    {
    }
}
