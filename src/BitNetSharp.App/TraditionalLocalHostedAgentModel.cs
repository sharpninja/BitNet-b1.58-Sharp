using BitNetSharp.Core;

namespace BitNetSharp.App;

public sealed class TraditionalLocalHostedAgentModel : IHostedAgentModel, ITrainableHostedAgentModel
{
    private const string DefaultFallbackResponse = "local model ready";

    private readonly BitNetTokenizer _tokenizer;
    private readonly Dictionary<string, string> _memorizedResponses = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Dictionary<string, int>> _transitions = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _priors = new(StringComparer.Ordinal);
    private readonly object _gate = new();
    private bool _isTrained;
    private string _mostFrequentPriorToken = string.Empty;

    public TraditionalLocalHostedAgentModel(VerbosityLevel verbosity)
    {
        Verbosity = verbosity;
        _tokenizer = new BitNetTokenizer(BitNetTrainingCorpus.CreateDefaultVocabulary());
    }

    public string AgentName => ModelId;

    public string ModelId => HostedAgentModelFactory.TraditionalLocalModelId;

    public string DisplayName => "Traditional local count-based language model";

    public string PrimaryLanguage => "en-US";

    public VerbosityLevel Verbosity { get; }

    public string SystemPrompt => "Respond in clear American English using the traditional local comparison model.";

    public IReadOnlyList<string> DescribeModel() =>
    [
        DisplayName,
        $"Model ID: {ModelId}",
        "Training: count-based next-token statistics over the default corpus",
        "Execution: in-process local comparator"
    ];

    public Task<HostedAgentModelResponse> GetResponseAsync(
        string prompt,
        int? maxOutputTokens = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        EnsureTrained();

        var normalizedPrompt = NormalizePromptKey(prompt);
        if (_memorizedResponses.TryGetValue(normalizedPrompt, out var memorizedResponse))
        {
            return Task.FromResult(CreateResponse(memorizedResponse));
        }

        var generated = new List<string>();
        var context = _tokenizer.Tokenize(prompt).LastOrDefault() ?? BitNetTokenizer.BeginToken;
        var remainingTokens = Math.Max(1, maxOutputTokens ?? 8);

        for (var step = 0; step < remainingTokens; step++)
        {
            var nextToken = PredictNextToken(context);
            if (string.IsNullOrWhiteSpace(nextToken))
            {
                break;
            }

            generated.Add(nextToken);
            context = nextToken;
        }

        if (generated.Count == 0)
        {
            generated.AddRange(DefaultFallbackResponse.Split(' ', StringSplitOptions.RemoveEmptyEntries));
        }

        return Task.FromResult(CreateResponse(_tokenizer.Detokenize(generated)));
    }

    public void Train(IEnumerable<TrainingExample> examples, int epochs = 1)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        lock (_gate)
        {
            _memorizedResponses.Clear();
            _transitions.Clear();
            _priors.Clear();

            for (var epoch = 0; epoch < Math.Max(1, epochs); epoch++)
            {
                foreach (var example in trainingSet)
                {
                    var promptTokens = _tokenizer.Tokenize(example.Prompt).ToArray();
                    var responseTokens = _tokenizer.Tokenize(example.Response).ToArray();
                    _memorizedResponses[NormalizePromptKey(example.Prompt)] = _tokenizer.Detokenize(responseTokens);

                    var context = promptTokens.LastOrDefault() ?? BitNetTokenizer.BeginToken;
                    foreach (var token in responseTokens)
                    {
                        AddTransition(context, token);
                        context = token;
                    }
                }
            }

            _mostFrequentPriorToken = _priors.Count == 0
                ? string.Empty
                : _priors.MaxBy(static pair => pair.Value).Key;
            _isTrained = true;
        }
    }

    public void Dispose()
    {
    }

    private void EnsureTrained()
    {
        if (_isTrained)
        {
            return;
        }

        Train(BitNetTrainingCorpus.CreateDefaultExamples(), epochs: 3);
    }

    private HostedAgentModelResponse CreateResponse(string text)
    {
        var diagnostics = Verbosity == VerbosityLevel.Quiet
            ? Array.Empty<string>()
            : new[]
            {
                $"Model: {ModelId}",
                "Architecture: traditional count-based language model",
                $"Primary language: {PrimaryLanguage}"
            };

        return new HostedAgentModelResponse(text, diagnostics);
    }

    private void AddTransition(string context, string token)
    {
        if (!_transitions.TryGetValue(context, out var counts))
        {
            counts = new Dictionary<string, int>(StringComparer.Ordinal);
            _transitions[context] = counts;
        }

        counts[token] = counts.TryGetValue(token, out var existing) ? existing + 1 : 1;
        _priors[token] = _priors.TryGetValue(token, out var prior) ? prior + 1 : 1;
    }

    private string PredictNextToken(string context)
    {
        if (!_transitions.TryGetValue(context, out var counts) || counts.Count == 0)
        {
            return _mostFrequentPriorToken;
        }

        var bestToken = string.Empty;
        var bestCount = -1;

        foreach (var pair in counts)
        {
            if (pair.Value > bestCount
                || (pair.Value == bestCount && string.CompareOrdinal(pair.Key, bestToken) < 0))
            {
                bestToken = pair.Key;
                bestCount = pair.Value;
            }
        }

        return bestToken;
    }

    private string NormalizePromptKey(string prompt) => string.Join(' ', _tokenizer.Tokenize(prompt));
}
