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

        _beginTokenId = _tokenToId[BitNetTokenizer.BeginToken];
        _endTokenId = _tokenToId[BitNetTokenizer.EndToken];
        _tokenizer = new BitNetTokenizer(_idToToken);

        Config = config ?? CreateDefaultConfig(_idToToken.Length);
        if (Config.VocabSize != _idToToken.Length)
        {
            throw new ArgumentException($"The BitNetConfig vocabulary size ({Config.VocabSize}) must match the tokenizer vocabulary size ({_idToToken.Length}).", nameof(config));
        }

        Transformer = new BitNetTransformer(Config, seed);
    }

    public BitNetOptions Options { get; }

    public BitNetConfig Config { get; }

    public BitNetTransformer Transformer { get; }

    public string ModelId => "bitnet-b1.58-sharp";

    public BitNetTokenizer Tokenizer => _tokenizer;

    public static BitNetPaperModel CreateDefault(VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        new(new BitNetOptions(BitNetTrainingCorpus.CreateDefaultVocabulary(), verbosity));

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
        var maxPredictionCount = Math.Min(Options.MaxResponseTokens, systemPredictionLimit);
        var requestedPredictionCount = maxTokens.GetValueOrDefault(maxPredictionCount);
        var predictionCount = Math.Clamp(requestedPredictionCount, 1, maxPredictionCount);
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
