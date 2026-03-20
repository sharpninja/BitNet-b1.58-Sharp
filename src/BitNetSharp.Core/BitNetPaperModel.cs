using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core;

public sealed class BitNetPaperModel
{
    private const int MaxPredictionLimit = 8;

    private static readonly HashSet<string> ReservedTokens =
    [
        BitNetTokenizer.BeginToken,
        BitNetTokenizer.EndToken,
        BitNetTokenizer.UnknownToken
    ];

    private readonly int _beginTokenId;
    private readonly int _endTokenId;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly string[] _idToToken;
    private readonly BitNetTokenizer _tokenizer;

    public BitNetPaperModel(BitNetOptions options, BitNetConfig? config = null, int seed = 42)
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
                .Where(token => !ReservedTokens.Contains(token))
                .Distinct(StringComparer.Ordinal)
        ];

        _tokenToId = _idToToken
            .Select((token, index) => new { token, index })
            .ToDictionary(item => item.token, item => item.index, StringComparer.Ordinal);

        if (_idToToken.Length <= ReservedTokens.Count)
        {
            throw new ArgumentException("Options.Vocabulary must include at least one non-special token for the paper model.", nameof(options));
        }

        _beginTokenId = _tokenToId[BitNetTokenizer.BeginToken];
        _endTokenId = _tokenToId[BitNetTokenizer.EndToken];
        _tokenizer = new BitNetTokenizer(_idToToken);

        Config = config ?? CreateDefaultConfig(_idToToken.Length);
        if (Config.VocabSize != _idToToken.Length)
        {
            throw new ArgumentException($"The BitNetConfig vocabulary size ({Config.VocabSize}) must match the tokenizer vocabulary size ({_idToToken.Length}).", nameof(config));
        }

        // Use a deterministic default so the seeded paper model stays stable in tests and CLI inspection.
        Transformer = new BitNetTransformer(Config, seed);
    }

    public BitNetOptions Options { get; }

    public BitNetConfig Config { get; }

    public BitNetTransformer Transformer { get; }

    public string ModelId => "bitnet-b1.58-sharp";

    public BitNetTokenizer Tokenizer => _tokenizer;

    public static BitNetPaperModel CreateDefault(VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        new(new BitNetOptions(BitNetTrainingCorpus.CreateDefaultVocabulary(), verbosity));

    public TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 3, float learningRate = 0.05f)
    {
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        var weights = ExportOutputHeadWeights();
        var lossHistory = new List<double>(epochs);

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var totalLoss = 0d;
            var observations = 0;

            foreach (var example in trainingSet)
            {
                var promptIds = EncodeTokenIds(example.Prompt);
                var targetIds = EncodeTokenIds(example.Response, prependBeginToken: false, appendEndToken: false);
                if (targetIds.Count == 0)
                {
                    continue;
                }

                var hiddenStates = ForwardHiddenStates(promptIds);
                var features = GetLastRow(hiddenStates);
                var targetId = targetIds[0];
                var probabilities = ComputeProbabilities(weights, features);

                totalLoss -= Math.Log(Math.Max(probabilities[targetId], 1e-9d));
                observations++;

                for (var tokenId = 0; tokenId < probabilities.Length; tokenId++)
                {
                    var gradient = probabilities[tokenId] - (tokenId == targetId ? 1d : 0d);
                    for (var dimension = 0; dimension < features.Length; dimension++)
                    {
                        weights[tokenId, dimension] -= (float)(learningRate * gradient * features[dimension]);
                    }
                }
            }

            ImportOutputHeadWeights(weights);
            weights = ExportOutputHeadWeights();
            lossHistory.Add(observations == 0 ? 0d : totalLoss / observations);
        }

        var stats = GetTernaryWeightStats();
        return new TrainingReport(
            lossHistory,
            trainingSet.Count * epochs,
            epochs,
            stats.NegativeCount,
            stats.ZeroCount,
            stats.PositiveCount);
    }

    public BitNetGenerationResult GenerateResponse(string prompt, int? maxTokens = null)
    {
        var diagnostics = new List<string>();
        var inputTokenIds = TokenizeToIds(prompt);
        var truncated = false;

        if (inputTokenIds.Count > Config.MaxSequenceLength)
        {
            inputTokenIds = inputTokenIds.Skip(inputTokenIds.Count - Config.MaxSequenceLength).ToArray();
            truncated = true;
        }

        if (Options.Verbosity >= VerbosityLevel.Normal)
        {
            diagnostics.Add($"Model: {ModelId}");
            diagnostics.Add($"Architecture: decoder-only transformer ({Config.LayerCount} layers, dim {Config.Dimension}, heads {Config.HeadCount})");
            diagnostics.Add($"Primary language: {Options.PrimaryLanguage}");

            if (truncated)
            {
                diagnostics.Add($"Prompt truncated to the last {Config.MaxSequenceLength} tokens to fit the configured context window.");
            }
        }

        var logits = Transformer.Forward(inputTokenIds);
        var availableTokenCount = _idToToken.Length - ReservedTokens.Count;
        var systemPredictionLimit = Math.Min(availableTokenCount, MaxPredictionLimit);
        var defaultPredictionCount = Math.Min(Options.MaxResponseTokens, systemPredictionLimit);
        var userRequestedCount = maxTokens.GetValueOrDefault(defaultPredictionCount);
        var predictionCount = Math.Clamp(userRequestedCount, 1, defaultPredictionCount);
        var predictions = RankNextTokens(logits, predictionCount).ToArray();

        if (Options.Verbosity == VerbosityLevel.Verbose)
        {
            foreach (var prediction in predictions)
            {
                diagnostics.Add($"Prediction: token={prediction.Token}, logit={prediction.Logit:0.###}");
            }
        }

        if (Options.Verbosity == VerbosityLevel.Quiet)
        {
            diagnostics.Clear();
        }

        var responseText = $"Top next-token predictions: {string.Join(", ", predictions.Select(prediction => prediction.Token))}.";
        return new BitNetGenerationResult(
            responseText,
            predictions.Select(prediction => prediction.Token).ToArray(),
            diagnostics);
    }

    public double CalculatePerplexity(IEnumerable<string> validationSamples)
    {
        ArgumentNullException.ThrowIfNull(validationSamples);

        var totalLoss = 0d;
        var totalTokens = 0;
        foreach (var sample in validationSamples)
        {
            var tokenIds = EncodeTokenIds(sample, appendEndToken: true);
            for (var index = 0; index < tokenIds.Count - 1; index++)
            {
                var context = tokenIds.Take(index + 1).ToArray();
                var logits = ForwardLogits(context);
                totalLoss -= Math.Log(GetTargetProbability(logits, tokenIds[index + 1]));
                totalTokens++;
            }
        }

        return totalTokens == 0 ? 0d : Math.Exp(totalLoss / totalTokens);
    }

    public TernaryWeightStats GetTernaryWeightStats()
    {
        var negative = 0;
        var zero = 0;
        var positive = 0;

        foreach (var layer in EnumerateBitLinearLayers())
        {
            var stats = layer.GetTernaryStats();
            negative += stats.NegativeCount;
            zero += stats.ZeroCount;
            positive += stats.PositiveCount;
        }

        return new TernaryWeightStats(negative, zero, positive);
    }

    internal IReadOnlyList<int> EncodeTokenIds(string text, bool prependBeginToken = true, bool appendEndToken = false)
    {
        var tokenIds = new List<int>();
        if (prependBeginToken)
        {
            tokenIds.Add(_beginTokenId);
        }

        tokenIds.AddRange(_tokenizer.Tokenize(text).Select(GetId));
        if (appendEndToken)
        {
            tokenIds.Add(_endTokenId);
        }

        return tokenIds;
    }

    internal float[,] ForwardLogits(IReadOnlyList<int> tokenIds) => Transformer.Forward(tokenIds);

    internal float[,] ForwardHiddenStates(IReadOnlyList<int> tokenIds) => Transformer.ForwardHiddenStates(tokenIds);

    internal float[,] ExportOutputHeadWeights() => Transformer.OutputHead.ToFullPrecision();

    internal void ImportOutputHeadWeights(float[,] weights) => Transformer.OutputHead.QuantizeFromFullPrecision(weights);

    private static double GetTargetProbability(float[,] logits, int targetId)
    {
        var lastRow = logits.GetLength(0) - 1;
        var maxLogit = double.NegativeInfinity;
        for (var column = 0; column < logits.GetLength(1); column++)
        {
            maxLogit = Math.Max(maxLogit, logits[lastRow, column]);
        }

        var partition = 0d;
        var targetProbability = 0d;
        for (var column = 0; column < logits.GetLength(1); column++)
        {
            var probabilityMass = Math.Exp(logits[lastRow, column] - maxLogit);
            partition += probabilityMass;
            if (column == targetId)
            {
                targetProbability = probabilityMass;
            }
        }

        if (partition <= 0d)
        {
            return 1e-9d;
        }

        return Math.Max(targetProbability / partition, 1e-9d);
    }

    private static BitNetConfig CreateDefaultConfig(int vocabularySize) =>
        new(
            vocabSize: vocabularySize,
            dimension: 256,
            hiddenDimension: 1_024,
            layerCount: 4,
            headCount: 8,
            maxSequenceLength: 256);

    private IReadOnlyList<int> TokenizeToIds(string prompt)
    {
        var tokenIds = new List<int> { _beginTokenId };
        tokenIds.AddRange(_tokenizer.Tokenize(prompt).Select(GetId));
        return tokenIds;
    }

    private static float[] GetLastRow(float[,] matrix)
    {
        var lastRowIndex = matrix.GetLength(0) - 1;
        var result = new float[matrix.GetLength(1)];
        for (var column = 0; column < result.Length; column++)
        {
            result[column] = matrix[lastRowIndex, column];
        }

        return result;
    }

    private static double[] ComputeProbabilities(float[,] weights, float[] features)
    {
        var logits = new double[weights.GetLength(0)];
        var maxLogit = double.NegativeInfinity;

        for (var row = 0; row < weights.GetLength(0); row++)
        {
            var value = 0d;
            for (var column = 0; column < weights.GetLength(1); column++)
            {
                value += weights[row, column] * features[column];
            }

            logits[row] = value;
            maxLogit = Math.Max(maxLogit, value);
        }

        var partition = 0d;
        for (var index = 0; index < logits.Length; index++)
        {
            logits[index] = Math.Exp(logits[index] - maxLogit);
            partition += logits[index];
        }

        if (partition <= 0d)
        {
            return Enumerable.Repeat(1d / logits.Length, logits.Length).ToArray();
        }

        for (var index = 0; index < logits.Length; index++)
        {
            logits[index] /= partition;
        }

        return logits;
    }

    private IEnumerable<(string Token, float Logit)> RankNextTokens(float[,] logits, int count)
    {
        var lastRow = logits.GetLength(0) - 1;
        return Enumerable.Range(0, logits.GetLength(1))
            .Where(id => id != _beginTokenId && id != _endTokenId && id != _tokenToId[BitNetTokenizer.UnknownToken])
            .OrderByDescending(id => logits[lastRow, id])
            .Take(count)
            .Select(id => (_idToToken[id], logits[lastRow, id]));
    }

    private IEnumerable<Layers.BitLinear> EnumerateBitLinearLayers()
    {
        foreach (var layer in Transformer.Layers)
        {
            yield return layer.Attention.QueryProjection;
            yield return layer.Attention.KeyProjection;
            yield return layer.Attention.ValueProjection;
            yield return layer.Attention.OutputProjection;
            yield return layer.FeedForward.GateProjection;
            yield return layer.FeedForward.UpProjection;
            yield return layer.FeedForward.DownProjection;
        }

        yield return Transformer.OutputHead;
    }

    private int GetId(string token) => _tokenToId.TryGetValue(token, out var id) ? id : _tokenToId[BitNetTokenizer.UnknownToken];
}
