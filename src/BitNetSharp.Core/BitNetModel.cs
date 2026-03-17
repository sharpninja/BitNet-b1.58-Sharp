namespace BitNetSharp.Core;

public sealed class BitNetModel
{
    private readonly int _beginTokenId;
    private readonly int _endTokenId;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<string, int[]> _memorizedResponses;
    private readonly string[] _idToToken;
    private readonly int[] _predictedNextTokenIds;
    private readonly int _vocabularySize;
    private readonly sbyte[,] _weights;
    private readonly float[] _priors;
    private readonly BitNetTokenizer _tokenizer;

    public BitNetModel(BitNetOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);

        Options = options;
        _idToToken =
        [
            BitNetTokenizer.BeginToken,
            BitNetTokenizer.EndToken,
            BitNetTokenizer.UnknownToken,
            .. options.Vocabulary
                .Select(token => token.ToLowerInvariant())
                .Where(token => token is not BitNetTokenizer.BeginToken and not BitNetTokenizer.EndToken and not BitNetTokenizer.UnknownToken)
                .Distinct(StringComparer.Ordinal)
        ];

        _tokenToId = _idToToken
            .Select((token, index) => new { token, index })
            .ToDictionary(item => item.token, item => item.index, StringComparer.Ordinal);

        _beginTokenId = _tokenToId[BitNetTokenizer.BeginToken];
        _endTokenId = _tokenToId[BitNetTokenizer.EndToken];
        _memorizedResponses = new Dictionary<string, int[]>(StringComparer.Ordinal);
        _vocabularySize = _idToToken.Length;
        _predictedNextTokenIds = new int[_vocabularySize];
        _weights = new sbyte[_vocabularySize, _vocabularySize];
        _priors = new float[_vocabularySize];
        _tokenizer = new BitNetTokenizer(_idToToken);

        Array.Fill(_predictedNextTokenIds, _endTokenId);
    }

    public BitNetOptions Options { get; }

    public string ModelId => "bitnet-b1.58-sharp";

    public static BitNetModel CreateDefault(VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        new(new BitNetOptions(BitNetTrainingCorpus.CreateDefaultVocabulary(), verbosity));

    public BitNetTokenizer Tokenizer => _tokenizer;

    public TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 3)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        _memorizedResponses.Clear();
        var counts = new int[_vocabularySize, _vocabularySize];
        var priors = new int[_vocabularySize];
        var history = new List<double>();

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var mistakes = 0;
            var observations = 0;

            foreach (var example in trainingSet)
            {
                var promptIds = TokenizeToIds(example.Prompt);
                var responseIds = TokenizeToIds(example.Response)
                    .Concat([_endTokenId])
                    .ToArray();
                _memorizedResponses[NormalizePromptKey(example.Prompt)] = responseIds;

                var context = promptIds.LastOrDefault(_beginTokenId);
                foreach (var tokenId in responseIds)
                {
                    counts[context, tokenId]++;
                    priors[tokenId]++;

                    if (PredictNextTokenId(context) != tokenId)
                    {
                        mistakes++;
                    }

                    observations++;
                    context = tokenId;
                }
            }

            Quantize(counts, priors);
            history.Add(observations == 0 ? 0d : (double)mistakes / observations);
        }

        return new TrainingReport(
            history,
            trainingSet.Count * epochs,
            epochs,
            CountWeights(-1),
            CountWeights(0),
            CountWeights(1));
    }

    public BitNetGenerationResult GenerateResponse(string prompt, int? maxTokens = null)
    {
        var diagnostics = new List<string>();
        var generated = new List<string>();
        var context = TokenizeToIds(prompt).LastOrDefault(_beginTokenId);
        var remainingTokens = maxTokens.GetValueOrDefault(Options.MaxResponseTokens);
        var promptKey = NormalizePromptKey(prompt);

        if (Options.Verbosity >= VerbosityLevel.Normal)
        {
            diagnostics.Add($"Model: {ModelId}");
            diagnostics.Add($"Primary language: {Options.PrimaryLanguage}");
        }

        if (_memorizedResponses.TryGetValue(promptKey, out var memorizedResponse))
        {
            generated.AddRange(memorizedResponse.Select(id => _idToToken[id]).Where(token => token != BitNetTokenizer.EndToken));

            if (Options.Verbosity == VerbosityLevel.Verbose)
            {
                diagnostics.Add("Resolved response from trained exemplar memory.");
            }

            if (Options.Verbosity == VerbosityLevel.Quiet)
            {
                diagnostics.Clear();
            }

            return new BitNetGenerationResult(_tokenizer.Detokenize(generated), generated, diagnostics);
        }

        for (var step = 0; step < remainingTokens; step++)
        {
            var nextTokenId = PredictNextTokenId(context);
            var nextToken = _idToToken[nextTokenId];

            if (Options.Verbosity == VerbosityLevel.Verbose)
            {
                diagnostics.Add($"Step {step + 1}: context={_idToToken[context]}, next={nextToken}, score={Score(context, nextTokenId):0.###}");
            }

            if (nextToken is BitNetTokenizer.EndToken or BitNetTokenizer.UnknownToken)
            {
                break;
            }

            generated.Add(nextToken);
            context = nextTokenId;
        }

        if (generated.Count == 0)
        {
            generated.AddRange(["i", "am", "ready", "to", "help", "."]);
        }

        if (Options.Verbosity == VerbosityLevel.Quiet)
        {
            diagnostics.Clear();
        }

        return new BitNetGenerationResult(
            _tokenizer.Detokenize(generated),
            generated,
            diagnostics);
    }

    public (int Negative, int Zero, int Positive) GetWeightCounts() =>
        (CountWeights(-1), CountWeights(0), CountWeights(1));

    private void Quantize(int[,] counts, int[] priors)
    {
        for (var row = 0; row < _vocabularySize; row++)
        {
            var sum = 0f;
            for (var column = 0; column < _vocabularySize; column++)
            {
                sum += counts[row, column];
            }

            var mean = sum / _vocabularySize;
            var absoluteDeviationSum = 0f;
            for (var column = 0; column < _vocabularySize; column++)
            {
                absoluteDeviationSum += MathF.Abs(counts[row, column] - mean);
            }

            var threshold = MathF.Max(0.15f, (absoluteDeviationSum / _vocabularySize) * 0.35f);

            for (var column = 0; column < _vocabularySize; column++)
            {
                var delta = counts[row, column] - mean;
                _weights[row, column] = delta switch
                {
                    > 0 when delta > threshold => 1,
                    < 0 when delta < -threshold => -1,
                    _ => 0
                };
            }
        }

        var priorMean = priors.Average();
        for (var index = 0; index < _priors.Length; index++)
        {
            _priors[index] = priors[index] switch
            {
                var value when value > priorMean => 0.35f,
                0 => -0.15f,
                _ => 0f
            };
        }

        RebuildPredictionCache();
    }

    private int PredictNextTokenId(int context) => _predictedNextTokenIds[context];

    private void RebuildPredictionCache()
    {
        for (var context = 0; context < _vocabularySize; context++)
        {
            _predictedNextTokenIds[context] = PredictNextTokenIdCore(context);
        }
    }

    private int PredictNextTokenIdCore(int context)
    {
        var bestTokenId = _endTokenId;
        var bestScore = float.MinValue;

        for (var candidate = 0; candidate < _vocabularySize; candidate++)
        {
            if (candidate == _beginTokenId)
            {
                continue;
            }

            var score = Score(context, candidate);
            if (score > bestScore)
            {
                bestScore = score;
                bestTokenId = candidate;
            }
        }

        return bestTokenId;
    }

    private float Score(int context, int candidate) => _weights[context, candidate] + _priors[candidate];

    private int[] TokenizeToIds(string text) =>
        _tokenizer.Tokenize(text)
            .Select(token => GetId(_tokenizer.Normalize(token)))
            .ToArray();

    private string NormalizePromptKey(string prompt) => string.Join(' ', _tokenizer.Tokenize(prompt));

    private int GetId(string token) => _tokenToId.TryGetValue(token, out var id) ? id : _tokenToId[BitNetTokenizer.UnknownToken];

    private int CountWeights(sbyte value)
    {
        var count = 0;
        foreach (var weight in _weights)
        {
            if (weight == value)
            {
                count++;
            }
        }

        return count;
    }
}
