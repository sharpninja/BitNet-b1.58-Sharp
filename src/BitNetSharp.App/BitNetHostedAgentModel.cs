using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.App;

public sealed class BitNetHostedAgentModel(BitNetPaperModel model) : IHostedAgentModel, IInspectableHostedAgentModel, ITrainableHostedAgentModel
{
    public BitNetPaperModel Model { get; } = model ?? throw new ArgumentNullException(nameof(model));

    public string AgentName => Model.ModelId;

    public string ModelId => Model.ModelId;

    public string DisplayName => "Paper-aligned BitNet b1.58 transformer";

    public string PrimaryLanguage => Model.Options.PrimaryLanguage;

    public VerbosityLevel Verbosity => Model.Options.Verbosity;

    public string SystemPrompt => "Respond in clear American English using the paper-aligned BitNet b1.58 transformer diagnostics.";

    public IReadOnlyList<string> DescribeModel() =>
    [
        DisplayName,
        $"Model ID: {ModelId}",
        $"Vocabulary size: {Model.Config.VocabSize}",
        $"Layers: {Model.Config.LayerCount}",
        $"Dimension: {Model.Config.Dimension}",
        $"Hidden dimension: {Model.Config.HiddenDimension}",
        $"Heads: {Model.Config.HeadCount}",
        $"Max sequence length: {Model.Config.MaxSequenceLength}"
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

    public TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 1)
    {
        return Model.Train(examples, epochs);
    }

    public void Dispose()
    {
    }
}
