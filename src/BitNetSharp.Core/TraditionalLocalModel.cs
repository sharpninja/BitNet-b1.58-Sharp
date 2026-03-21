using System.Numerics.Tensors;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core;

public sealed class TraditionalLocalModel
{
    public const int DefaultTrainingEpochs = 24;

    private const int DefaultEmbeddingDimension = 48;
    private const int DefaultContextWindow = 8;
    private const float DefaultLearningRate = 0.15f;
    private const float MinimumProbability = 1e-6f;

    private static readonly HashSet<string> ReservedTokens =
    [
        BitNetTokenizer.BeginToken,
        BitNetTokenizer.EndToken,
        BitNetTokenizer.UnknownToken
    ];

    private readonly int _beginTokenId;
    private readonly int _endTokenId;
    private readonly int _unknownTokenId;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly string[] _idToToken;
    private readonly float[] _tokenEmbeddings;
    private readonly float[] _outputWeights;
    private readonly float[] _outputBias;
    private readonly BitNetTokenizer _tokenizer;
    private readonly object _gate = new();
    private readonly int _seed;
    private bool _isTrained;

    public TraditionalLocalModel(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        int embeddingDimension = DefaultEmbeddingDimension,
        int contextWindow = DefaultContextWindow,
        int seed = 7)
        : this(
            new BitNetOptions(BitNetTrainingCorpus.CreateVocabulary(trainingExamples), verbosity),
            embeddingDimension,
            contextWindow,
            seed)
    {
    }

    public TraditionalLocalModel(
        BitNetOptions options,
        int embeddingDimension = DefaultEmbeddingDimension,
        int contextWindow = DefaultContextWindow,
        int seed = 7)
    {
        ArgumentNullException.ThrowIfNull(options);

        if (embeddingDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embeddingDimension), "Embedding dimension must be positive.");
        }

        if (contextWindow <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(contextWindow), "Context window must be positive.");
        }

        Options = options;
        EmbeddingDimension = embeddingDimension;
        ContextWindow = contextWindow;
        _seed = seed;

        _idToToken =
        [
            BitNetTokenizer.BeginToken,
            BitNetTokenizer.EndToken,
            BitNetTokenizer.UnknownToken,
            .. options.Vocabulary
                .Select(token => token.ToLowerInvariant())
                .Where(token => !ReservedTokens.Contains(token))
                .Distinct(StringComparer.Ordinal)
        ];

        _tokenToId = _idToToken
            .Select((token, index) => new { token, index })
            .ToDictionary(item => item.token, item => item.index, StringComparer.Ordinal);

        _beginTokenId = _tokenToId[BitNetTokenizer.BeginToken];
        _endTokenId = _tokenToId[BitNetTokenizer.EndToken];
        _unknownTokenId = _tokenToId[BitNetTokenizer.UnknownToken];
        _tokenizer = new BitNetTokenizer(_idToToken);
        _tokenEmbeddings = new float[_idToToken.Length * EmbeddingDimension];
        _outputWeights = new float[_idToToken.Length * InputDimension];
        _outputBias = new float[_idToToken.Length];

        ResetParameters();
    }

    public BitNetOptions Options { get; }

    public string ModelId => "traditional-local";

    public int EmbeddingDimension { get; }

    public int ContextWindow { get; }

    public int InputDimension => ContextWindow * EmbeddingDimension;

    public BitNetTokenizer Tokenizer => _tokenizer;

    public long EstimateResidentParameterBytes() =>
        ((long)_tokenEmbeddings.Length + _outputWeights.Length + _outputBias.Length) * sizeof(float);

    internal int Seed => _seed;

    public static TraditionalLocalModel CreateDefault(VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        new(new BitNetOptions(BitNetTrainingCorpus.CreateDefaultVocabulary(), verbosity));

    public static TraditionalLocalModel CreateForTrainingCorpus(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        new(trainingExamples, verbosity);

    public TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 3, float learningRate = DefaultLearningRate)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        if (epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs), "Epochs must be positive.");
        }

        if (learningRate <= 0f || float.IsNaN(learningRate) || float.IsInfinity(learningRate))
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be a positive finite value.");
        }

        lock (_gate)
        {
            ResetParameters();

            var history = new List<double>(epochs);
            var totalSamples = 0;
            var hidden = new float[InputDimension];
            var logits = new float[_idToToken.Length];
            var probabilities = new float[_idToToken.Length];
            var hiddenGradient = new float[InputDimension];

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                double epochLoss = 0d;
                var epochSamples = 0;

                foreach (var example in trainingSet)
                {
                    var context = TokenizeToIds(example.Prompt).TakeLast(ContextWindow).ToList();
                    var responseIds = TokenizeToIds(example.Response)
                        .Skip(1)
                        .Concat([_endTokenId])
                        .ToArray();

                    foreach (var targetTokenId in responseIds)
                    {
                        epochLoss += TrainStep(context, targetTokenId, learningRate, hidden, logits, probabilities, hiddenGradient);
                        epochSamples++;
                        context.Add(targetTokenId);

                        if (context.Count > ContextWindow)
                        {
                            context.RemoveAt(0);
                        }
                    }
                }

                totalSamples += epochSamples;
                history.Add(epochSamples == 0 ? 0d : epochLoss / epochSamples);
            }

            _isTrained = true;
            var stats = GetTernaryWeightStats();
            return new TrainingReport(
                history,
                totalSamples,
                epochs,
                stats.NegativeCount,
                stats.ZeroCount,
                stats.PositiveCount);
        }
    }

    public BitNetGenerationResult GenerateResponse(string prompt, int? maxTokens = null)
    {
        EnsureTrained();

        lock (_gate)
        {
            var diagnostics = new List<string>();
            if (Options.Verbosity >= VerbosityLevel.Normal)
            {
                diagnostics.Add($"Model: {ModelId}");
                diagnostics.Add($"Architecture: tensor-based ordered-context language model (embedding dim {EmbeddingDimension}, context {ContextWindow})");
                diagnostics.Add($"Primary language: {Options.PrimaryLanguage}");
            }

            var context = TokenizeToIds(prompt).TakeLast(ContextWindow).ToList();
            var generatedTokenIds = new List<int>();
            var maxGeneratedTokens = Math.Max(1, maxTokens.GetValueOrDefault(Options.MaxResponseTokens));
            var hidden = new float[InputDimension];
            var logits = new float[_idToToken.Length];
            var probabilities = new float[_idToToken.Length];

            for (var step = 0; step < maxGeneratedTokens; step++)
            {
                BuildHiddenState(context, hidden);
                ComputeProbabilities(hidden, logits, probabilities);

                var nextTokenId = SelectNextToken(probabilities, allowEndToken: generatedTokenIds.Count > 0);
                if (nextTokenId == _endTokenId)
                {
                    break;
                }

                generatedTokenIds.Add(nextTokenId);
                context.Add(nextTokenId);

                if (context.Count > ContextWindow)
                {
                    context.RemoveAt(0);
                }

                if (Options.Verbosity == VerbosityLevel.Verbose)
                {
                    diagnostics.Add($"Prediction: token={_idToToken[nextTokenId]}, probability={probabilities[nextTokenId]:0.###}");
                }
            }

            if (Options.Verbosity == VerbosityLevel.Quiet)
            {
                diagnostics.Clear();
            }

            var generatedTokens = generatedTokenIds.Select(id => _idToToken[id]).ToArray();
            var responseText = generatedTokens.Length == 0
                ? "Traditional local model is ready."
                : _tokenizer.Detokenize(generatedTokens);

            return new BitNetGenerationResult(responseText, generatedTokens, diagnostics);
        }
    }

    public double CalculatePerplexity(IEnumerable<string> validationSamples)
    {
        ArgumentNullException.ThrowIfNull(validationSamples);
        EnsureTrained();

        lock (_gate)
        {
            var totalLoss = 0d;
            var totalTokens = 0;
            var hidden = new float[InputDimension];
            var logits = new float[_idToToken.Length];
            var probabilities = new float[_idToToken.Length];

            foreach (var sample in validationSamples)
            {
                var tokenIds = TokenizeToIds(sample).Concat([_endTokenId]).ToArray();
                for (var index = 0; index < tokenIds.Length - 1; index++)
                {
                    BuildHiddenState(tokenIds.Take(index + 1).TakeLast(ContextWindow).ToArray(), hidden);
                    ComputeProbabilities(hidden, logits, probabilities);
                    totalLoss -= Math.Log(Math.Max(probabilities[tokenIds[index + 1]], MinimumProbability));
                    totalTokens++;
                }
            }

            return totalTokens == 0 ? 0d : Math.Exp(totalLoss / totalTokens);
        }
    }

    public TernaryWeightStats GetTernaryWeightStats()
    {
        lock (_gate)
        {
            var negative = 0;
            var zero = 0;
            var positive = 0;

            CountWeightSigns(_tokenEmbeddings, ref negative, ref zero, ref positive);
            CountWeightSigns(_outputWeights, ref negative, ref zero, ref positive);
            CountWeightSigns(_outputBias, ref negative, ref zero, ref positive);

            return new TernaryWeightStats(negative, zero, positive);
        }
    }

    internal float[] ExportTokenEmbeddings()
    {
        lock (_gate)
        {
            return [.. _tokenEmbeddings];
        }
    }

    internal float[] ExportOutputWeights()
    {
        lock (_gate)
        {
            return [.. _outputWeights];
        }
    }

    internal float[] ExportOutputBias()
    {
        lock (_gate)
        {
            return [.. _outputBias];
        }
    }

    internal void ImportState(float[] tokenEmbeddings, float[] outputWeights, float[] outputBias)
    {
        ArgumentNullException.ThrowIfNull(tokenEmbeddings);
        ArgumentNullException.ThrowIfNull(outputWeights);
        ArgumentNullException.ThrowIfNull(outputBias);

        lock (_gate)
        {
            if (tokenEmbeddings.Length != _tokenEmbeddings.Length)
            {
                throw new ArgumentException($"Token embedding length {tokenEmbeddings.Length} does not match expected length {_tokenEmbeddings.Length}.", nameof(tokenEmbeddings));
            }

            if (outputWeights.Length != _outputWeights.Length)
            {
                throw new ArgumentException($"Output weight length {outputWeights.Length} does not match expected length {_outputWeights.Length}.", nameof(outputWeights));
            }

            if (outputBias.Length != _outputBias.Length)
            {
                throw new ArgumentException($"Output bias length {outputBias.Length} does not match expected length {_outputBias.Length}.", nameof(outputBias));
            }

            tokenEmbeddings.CopyTo(_tokenEmbeddings, 0);
            outputWeights.CopyTo(_outputWeights, 0);
            outputBias.CopyTo(_outputBias, 0);
            _isTrained = true;
        }
    }

    private void EnsureTrained()
    {
        if (_isTrained)
        {
            return;
        }

        Train(BitNetTrainingCorpus.CreateDefaultExamples(), epochs: DefaultTrainingEpochs);
    }

    private double TrainStep(
        IReadOnlyList<int> contextTokenIds,
        int targetTokenId,
        float learningRate,
        float[] hidden,
        float[] logits,
        float[] probabilities,
        float[] hiddenGradient)
    {
        BuildHiddenState(contextTokenIds, hidden);
        ComputeProbabilities(hidden, logits, probabilities);
        var targetProbability = MathF.Max(probabilities[targetTokenId], MinimumProbability);

        probabilities[targetTokenId] -= 1f;
        Array.Clear(hiddenGradient);

        for (var tokenId = 0; tokenId < _idToToken.Length; tokenId++)
        {
            var outputRow = GetOutputWeightSpan(tokenId);
            var gradient = probabilities[tokenId];
            if (gradient == 0f)
            {
                continue;
            }

            for (var dimension = 0; dimension < InputDimension; dimension++)
            {
                hiddenGradient[dimension] += gradient * outputRow[dimension];
            }
        }

        for (var tokenId = 0; tokenId < _idToToken.Length; tokenId++)
        {
            var outputRow = GetOutputWeightSpan(tokenId);
            var gradient = probabilities[tokenId];
            if (gradient == 0f)
            {
                continue;
            }

            for (var dimension = 0; dimension < InputDimension; dimension++)
            {
                outputRow[dimension] -= learningRate * gradient * hidden[dimension];
            }

            _outputBias[tokenId] -= learningRate * gradient;
        }

        var paddedContext = PadContext(contextTokenIds);
        for (var position = 0; position < paddedContext.Length; position++)
        {
            var embedding = GetEmbeddingSpan(paddedContext[position]);
            var gradientOffset = position * EmbeddingDimension;
            for (var dimension = 0; dimension < EmbeddingDimension; dimension++)
            {
                embedding[dimension] -= learningRate * hiddenGradient[gradientOffset + dimension];
            }
        }

        return -Math.Log(targetProbability);
    }

    private void BuildHiddenState(IReadOnlyList<int> contextTokenIds, float[] hidden)
    {
        var paddedContext = PadContext(contextTokenIds);
        for (var position = 0; position < paddedContext.Length; position++)
        {
            var embedding = GetReadOnlyEmbeddingSpan(paddedContext[position]);
            var hiddenOffset = position * EmbeddingDimension;
            for (var dimension = 0; dimension < EmbeddingDimension; dimension++)
            {
                hidden[hiddenOffset + dimension] = embedding[dimension];
            }
        }
    }

    private void ComputeProbabilities(float[] hidden, float[] logits, float[] probabilities)
    {
        var maxLogit = float.NegativeInfinity;

        for (var tokenId = 0; tokenId < _idToToken.Length; tokenId++)
        {
            var logit = TensorPrimitives.Dot(hidden, GetReadOnlyOutputWeightSpan(tokenId)) + _outputBias[tokenId];
            logits[tokenId] = logit;
            maxLogit = MathF.Max(maxLogit, logit);
        }

        for (var tokenId = 0; tokenId < logits.Length; tokenId++)
        {
            logits[tokenId] -= maxLogit;
        }

        TensorPrimitives.SoftMax(logits, probabilities);
    }

    private int SelectNextToken(float[] probabilities, bool allowEndToken)
    {
        var bestTokenId = _endTokenId;
        var bestProbability = float.NegativeInfinity;

        for (var tokenId = 0; tokenId < probabilities.Length; tokenId++)
        {
            if (tokenId == _beginTokenId || tokenId == _unknownTokenId || (!allowEndToken && tokenId == _endTokenId))
            {
                continue;
            }

            var probability = probabilities[tokenId];
            if (probability > bestProbability)
            {
                bestTokenId = tokenId;
                bestProbability = probability;
            }
        }

        return bestTokenId;
    }

    private IReadOnlyList<int> TokenizeToIds(string text)
    {
        var tokenIds = new List<int> { _beginTokenId };
        tokenIds.AddRange(_tokenizer.Tokenize(text).Select(GetId));
        return tokenIds;
    }

    private int GetId(string token) => _tokenToId.TryGetValue(token, out var id) ? id : _unknownTokenId;

    private Span<float> GetEmbeddingSpan(int tokenId) => _tokenEmbeddings.AsSpan(tokenId * EmbeddingDimension, EmbeddingDimension);

    private ReadOnlySpan<float> GetReadOnlyEmbeddingSpan(int tokenId) => _tokenEmbeddings.AsSpan(tokenId * EmbeddingDimension, EmbeddingDimension);

    private Span<float> GetOutputWeightSpan(int tokenId) => _outputWeights.AsSpan(tokenId * InputDimension, InputDimension);

    private ReadOnlySpan<float> GetReadOnlyOutputWeightSpan(int tokenId) => _outputWeights.AsSpan(tokenId * InputDimension, InputDimension);

    private void ResetParameters()
    {
        Array.Clear(_outputBias);

        var random = new Random(_seed);
        FillWithDeterministicNoise(_tokenEmbeddings, random);
        FillWithDeterministicNoise(_outputWeights, random);
        _isTrained = false;
    }

    private static void CountWeightSigns(float[] values, ref int negative, ref int zero, ref int positive)
    {
        foreach (var value in values)
        {
            if (value > 0f)
            {
                positive++;
            }
            else if (value < 0f)
            {
                negative++;
            }
            else
            {
                zero++;
            }
        }
    }

    private static void FillWithDeterministicNoise(float[] values, Random random)
    {
        for (var index = 0; index < values.Length; index++)
        {
            values[index] = ((float)random.NextDouble() - 0.5f) * 0.1f;
        }
    }

    private int[] PadContext(IReadOnlyList<int> contextTokenIds)
    {
        var paddedContext = Enumerable.Repeat(_beginTokenId, ContextWindow).ToArray();
        var sourceStart = Math.Max(0, contextTokenIds.Count - ContextWindow);
        var copyLength = Math.Min(contextTokenIds.Count, ContextWindow);

        for (var index = 0; index < copyLength; index++)
        {
            paddedContext[ContextWindow - copyLength + index] = contextTokenIds[sourceStart + index];
        }

        return paddedContext;
    }
}
