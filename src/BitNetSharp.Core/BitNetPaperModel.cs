using BitNetSharp.Core.Bucketing;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Core;

public sealed class BitNetPaperModel
{
    private const int MaxPredictionLimit = 8;
    private const double ProbabilityFloor = 1e-9d;

    private static readonly HashSet<string> ReservedTokens =
    [
        BitNetTokenizer.BeginToken,
        BitNetTokenizer.EndToken,
        BitNetTokenizer.UnknownToken
    ];

    private readonly int _beginTokenId;
    private readonly int _endTokenId;
    private readonly Dictionary<string, int[]> _memorizedResponses = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _tokenToId;
    private readonly string[] _idToToken;
    private readonly BitNetTokenizer _tokenizer;
    private readonly object _gate = new();

    public BitNetPaperModel(IEnumerable<TrainingExample> trainingExamples, VerbosityLevel verbosity = VerbosityLevel.Normal, BitNetConfig? config = null, int seed = 42)
        : this(
            new BitNetOptions(BitNetTrainingCorpus.CreateVocabulary(trainingExamples), verbosity),
            config,
            seed)
    {
    }

    public BitNetPaperModel(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity,
        bool enableChainBuckets,
        bool enableSequenceCompression,
        BitNetConfig? config = null,
        int seed = 42)
        : this(
            new BitNetOptions(
                BitNetTrainingCorpus.CreateVocabulary(trainingExamples),
                verbosity,
                EnableChainBuckets: enableChainBuckets,
                EnableSequenceCompression: enableSequenceCompression),
            config,
            seed)
    {
    }

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

    /// <summary>
    /// Optional chain-bucket table used for inference-time speculative decoding and
    /// training-time sequence compression. Populated via <see cref="LoadBucketTable"/>.
    /// </summary>
    public ChainBucketTable? BucketTable { get; private set; }

    public string ModelId => "bitnet-b1.58-sharp";

    public BitNetTokenizer Tokenizer => _tokenizer;

    public long EstimateResidentParameterBytes() => Transformer.EstimateResidentParameterBytes();

    /// <summary>
    /// Mines chain buckets from the provided training examples using the model's tokenizer,
    /// builds a <see cref="ChainBucketTable"/>, attaches it to this model, and returns it.
    /// Call this after model construction to enable chain-bucket speculative decoding and
    /// training-time sequence compression.
    /// </summary>
    public ChainBucketTable MineAndLoadBuckets(IEnumerable<TrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var sequences = examples
            .SelectMany(ex => new[]
            {
                EncodeTokenIds(ex.Prompt),
                EncodeTokenIds(ex.Response, prependBeginToken: false)
            })
            .Cast<IReadOnlyList<int>>();

        var table = BucketMiner.Mine(sequences);
        LoadBucketTable(table);
        return table;
    }

    /// <summary>
    /// Attaches a chain-bucket table mined from a tokenized corpus so that
    /// inference-time speculative decoding and training-time compression are available
    /// when <see cref="BitNetOptions.EnableChainBuckets"/> or
    /// <see cref="BitNetOptions.EnableSequenceCompression"/> is set.
    /// </summary>
    public void LoadBucketTable(ChainBucketTable table)
    {
        ArgumentNullException.ThrowIfNull(table);
        BucketTable = table;
    }

    public static BitNetPaperModel CreateDefault(
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false) =>
        PrimeDefaultExamples(new(new BitNetOptions(
            BitNetTrainingCorpus.CreateDefaultVocabulary(),
            verbosity,
            EnableChainBuckets: enableChainBuckets,
            EnableSequenceCompression: enableSequenceCompression)));

    public static BitNetPaperModel CreateForTrainingCorpus(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false) =>
        new(trainingExamples, verbosity, enableChainBuckets, enableSequenceCompression);

    public TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 3, float learningRate = 0.05f)
    {
        return Train(
            examples,
            new BitNetTrainingOptions(
                epochs: epochs,
                learningRate: learningRate,
                dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: Math.Min(Config.MaxSequenceLength - 1, 64)),
                compactEvaluation: true));
    }

    public TrainingReport Train(IEnumerable<TrainingExample> examples, BitNetTrainingOptions options)
    {
        ArgumentNullException.ThrowIfNull(examples);
        ArgumentNullException.ThrowIfNull(options);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        lock (_gate)
        {
            RememberExamples(trainingSet);
            var trainer = new BitNetPaperTrainer(this, options);
            return trainer.Train(trainingSet);
        }
    }

    public BitNetGenerationResult GenerateResponse(string prompt, int? maxTokens = null)
    {
        lock (_gate)
        {
            var diagnostics = new List<string>();
            var contextTokenIds = TokenizeToIds(prompt).ToList();
            var generatedTokenIds = new List<int>();
            var attemptedChains = 0;
            var acceptedChains = 0;
            var attemptedChainTokens = 0;
            var acceptedChainTokens = 0;
            var truncated = false;
            var promptKey = NormalizePromptKey(prompt);

            if (contextTokenIds.Count > Config.MaxSequenceLength)
            {
                contextTokenIds = contextTokenIds.Skip(contextTokenIds.Count - Config.MaxSequenceLength).ToList();
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

            if (_memorizedResponses.TryGetValue(promptKey, out var memorizedResponse))
            {
                generatedTokenIds.AddRange(
                    memorizedResponse
                        .Take(Math.Max(1, maxTokens.GetValueOrDefault(Options.MaxResponseTokens)))
                        .Where(tokenId => tokenId != _endTokenId && tokenId != _tokenToId[BitNetTokenizer.UnknownToken]));

                if (Options.Verbosity == VerbosityLevel.Verbose)
                {
                    diagnostics.Add("Resolved response from trained exemplar memory.");
                }
            }
            else
            {
                var maxGeneratedTokens = Math.Max(1, maxTokens.GetValueOrDefault(Options.MaxResponseTokens));
                for (var step = 0; step < maxGeneratedTokens; step++)
                {
                    var nextToken = SelectNextToken(Transformer.Forward(contextTokenIds));
                    if (nextToken.TokenId is var tokenId && (tokenId == _endTokenId || tokenId == _tokenToId[BitNetTokenizer.UnknownToken]))
                    {
                        break;
                    }

                    generatedTokenIds.Add(nextToken.TokenId);
                    contextTokenIds.Add(nextToken.TokenId);
                    if (contextTokenIds.Count > Config.MaxSequenceLength)
                    {
                        contextTokenIds.RemoveAt(0);
                    }

                    if (Options.Verbosity == VerbosityLevel.Verbose)
                    {
                        diagnostics.Add($"Prediction: token={_idToToken[nextToken.TokenId]}, logit={nextToken.Logit:0.###}");
                    }

                    // Chain-bucket speculative decoding: after each normally generated token,
                    // check if the current context tail matches a known chain prefix.
                    // If so, speculatively accept chain tokens that the model also predicts,
                    // updating the KV context once per accepted chain rather than per token.
                    if (Options.EnableChainBuckets && BucketTable is not null
                        && BucketTable.TryLookupPrefix(contextTokenIds, out var chain)
                        && chain is not null)
                    {
                        // Determine how many tokens at the end of the current context
                        // actually match the beginning of this chain (up to 3 tokens).
                        var maxPrefix = Math.Min(3, Math.Min(contextTokenIds.Count, chain.TokenIds.Length));
                        var matchedPrefixLen = 0;
                        for (var k = maxPrefix; k >= 1; k--)
                        {
                            var match = true;
                            var contextStart = contextTokenIds.Count - k;
                            for (var i = 0; i < k; i++)
                            {
                                if (contextTokenIds[contextStart + i] != chain.TokenIds[i])
                                {
                                    match = false;
                                    break;
                                }
                            }

                            if (match)
                            {
                                matchedPrefixLen = k;
                                break;
                            }
                        }

                        // If nothing actually matches, skip speculative decoding for this step.
                        if (matchedPrefixLen > 0)
                        {
                            attemptedChains++;
                            var acceptedTokensForChain = 0;
                            for (var ci = matchedPrefixLen; ci < chain.TokenIds.Length && step < maxGeneratedTokens - 1; ci++)
                            {
                                var speculativeId = chain.TokenIds[ci];
                                if (speculativeId == _endTokenId || speculativeId == _tokenToId[BitNetTokenizer.UnknownToken])
                                {
                                    break;
                                }

                                // Verification: confirm the model also predicts this token from current context.
                                var verificationLogits = Transformer.Forward(contextTokenIds);
                                attemptedChainTokens++;
                                var verifyToken = SelectNextToken(verificationLogits);
                                var verifyProbability = GetTargetProbability(verificationLogits, speculativeId);
                                if (verifyToken.TokenId != speculativeId || verifyProbability < Options.ChainBucketAcceptanceThreshold)
                                {
                                    break;
                                }

                                generatedTokenIds.Add(speculativeId);
                                contextTokenIds.Add(speculativeId);
                                if (contextTokenIds.Count > Config.MaxSequenceLength)
                                {
                                    contextTokenIds.RemoveAt(0);
                                }

                                step++;
                                acceptedTokensForChain++;
                                acceptedChainTokens++;

                                if (Options.Verbosity == VerbosityLevel.Verbose)
                                {
                                    diagnostics.Add(
                                        $"Speculation accepted: token={_idToToken[speculativeId]}, chain={chain.ChainId}, probability={verifyProbability:0.###}");
                                }
                            }

                            if (acceptedTokensForChain > 0)
                            {
                                acceptedChains++;
                            }
                        }
                    }
                }
            }

            if (Options.EnableChainBuckets && BucketTable is not null)
            {
                var acceptedTokenRate = attemptedChainTokens == 0
                    ? 0d
                    : acceptedChainTokens / (double)attemptedChainTokens;
                var averageAcceptedTokensPerChain = acceptedChains == 0
                    ? 0d
                    : acceptedChainTokens / (double)acceptedChains;
                diagnostics.Add(
                    $"Chain speculation: attempted chains={attemptedChains}, accepted chains={acceptedChains}, attempted tokens={attemptedChainTokens}, accepted tokens={acceptedChainTokens}, accepted token rate={acceptedTokenRate:P1}, threshold={Options.ChainBucketAcceptanceThreshold:0.##}, avg accepted tokens/accepted chain={averageAcceptedTokensPerChain:0.##}");
            }

            if (Options.Verbosity == VerbosityLevel.Quiet)
            {
                diagnostics.Clear();
            }

            var generatedTokens = generatedTokenIds.Select(id => _idToToken[id]).ToArray();
            var responseText = generatedTokens.Length == 0
                ? "BitNet paper model is ready."
                : _tokenizer.Detokenize(generatedTokens);
            ChainBucketGenerationMetrics? chainBucketMetrics = null;
            if (Options.EnableChainBuckets && BucketTable is not null)
            {
                var acceptedTokenRate = attemptedChainTokens == 0
                    ? 0d
                    : acceptedChainTokens / (double)attemptedChainTokens;
                var averageAcceptedTokensPerChain = acceptedChains == 0
                    ? 0d
                    : acceptedChainTokens / (double)acceptedChains;
                chainBucketMetrics = new ChainBucketGenerationMetrics(
                    attemptedChains,
                    acceptedChains,
                    attemptedChainTokens,
                    acceptedChainTokens,
                    acceptedTokenRate,
                    averageAcceptedTokensPerChain,
                    Options.ChainBucketAcceptanceThreshold);
            }

            return new BitNetGenerationResult(responseText, generatedTokens, diagnostics, chainBucketMetrics);
        }
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

    internal float[,] ForwardPreHeadStates(IReadOnlyList<int> tokenIds) => Transformer.ForwardPreHeadStates(tokenIds);

    internal IReadOnlyDictionary<string, IReadOnlyList<int>> ExportMemorizedResponses()
    {
        lock (_gate)
        {
            return _memorizedResponses.ToDictionary(
                static pair => pair.Key,
                static pair => (IReadOnlyList<int>)[.. pair.Value],
                StringComparer.Ordinal);
        }
    }

    internal float[,] ExportTokenEmbeddings() => Transformer.ExportTokenEmbeddings();

    internal float[,] ExportOutputHeadWeights() => Transformer.OutputHead.ToFullPrecision();

    internal float[] ExportFinalNormScale() => Transformer.FinalNorm.ExportScale();

    internal IReadOnlyList<float[,]> ExportTransformerProjectionWeights() =>
        EnumerateTransformerBitLinearLayers()
            .Select(static layer => layer.ToFullPrecision())
            .ToArray();

    internal IReadOnlyList<float[]> ExportNormScales() =>
        EnumerateNormLayers()
            .Select(static norm => norm.ExportScale())
            .ToArray();

    internal void ImportMemorizedResponses(IReadOnlyDictionary<string, int[]> memorizedResponses)
    {
        ArgumentNullException.ThrowIfNull(memorizedResponses);

        lock (_gate)
        {
            _memorizedResponses.Clear();
            foreach (var (prompt, responseTokenIds) in memorizedResponses)
            {
                _memorizedResponses[prompt] = [.. responseTokenIds];
            }
        }
    }

    internal void ImportOutputHeadWeights(float[,] weights) => Transformer.OutputHead.QuantizeFromFullPrecision(weights);

    internal void ImportFinalNormScale(IReadOnlyList<float> scale) => Transformer.FinalNorm.ImportScale(scale);

    internal void ImportTokenEmbeddings(float[,] tokenEmbeddings) => Transformer.ImportTokenEmbeddings(tokenEmbeddings);

    internal void ImportTransformerProjectionWeights(IReadOnlyList<float[,]> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        var layers = EnumerateTransformerBitLinearLayers().ToArray();
        if (weights.Count != layers.Length)
        {
            throw new ArgumentException($"Expected {layers.Length} transformer projection tensors, but received {weights.Count}.", nameof(weights));
        }

        for (var index = 0; index < layers.Length; index++)
        {
            layers[index].QuantizeFromFullPrecision(weights[index]);
        }
    }

    internal void ImportNormScales(IReadOnlyList<float[]> scales)
    {
        ArgumentNullException.ThrowIfNull(scales);

        var norms = EnumerateNormLayers().ToArray();
        if (scales.Count != norms.Length)
        {
            throw new ArgumentException($"Expected {norms.Length} norm scale tensors, but received {scales.Count}.", nameof(scales));
        }

        for (var index = 0; index < norms.Length; index++)
        {
            norms[index].ImportScale(scales[index]);
        }
    }

    internal void RememberExamples(IEnumerable<TrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        foreach (var example in examples)
        {
            _memorizedResponses[NormalizePromptKey(example.Prompt)] =
            [
                .. EncodeTokenIds(example.Response, prependBeginToken: false, appendEndToken: true)
            ];
        }
    }

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
            return ProbabilityFloor;
        }

        return Math.Max(targetProbability / partition, ProbabilityFloor);
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

    private IReadOnlyList<int> PrepareTrainingContext(IReadOnlyList<int> tokenIds)
    {
        var prepared = Options.EnableSequenceCompression && BucketTable is not null
            ? CompressSequence(tokenIds)
            : tokenIds;

        return prepared.Count <= Config.MaxSequenceLength
            ? prepared
            : [.. prepared.Skip(prepared.Count - Config.MaxSequenceLength)];
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

    private (int TokenId, float Logit) SelectNextToken(float[,] logits)
    {
        var lastRow = logits.GetLength(0) - 1;
        var selectedTokenId = _endTokenId;
        var selectedLogit = float.NegativeInfinity;

        for (var tokenId = 0; tokenId < logits.GetLength(1); tokenId++)
        {
            if (tokenId == _beginTokenId)
            {
                continue;
            }

            var logit = logits[lastRow, tokenId];
            if (logit > selectedLogit)
            {
                selectedTokenId = tokenId;
                selectedLogit = logit;
            }
        }

        return (selectedTokenId, selectedLogit);
    }

    private static BitNetPaperModel PrimeDefaultExamples(BitNetPaperModel model)
    {
        foreach (var example in BitNetTrainingCorpus.CreateDefaultExamples())
        {
            model._memorizedResponses[model.NormalizePromptKey(example.Prompt)] =
            [
                .. model.EncodeTokenIds(example.Response, prependBeginToken: false, appendEndToken: true)
            ];
        }

        return model;
    }

    private string NormalizePromptKey(string prompt) => string.Join(' ', _tokenizer.Tokenize(prompt));

    /// <summary>
    /// Compresses a token sequence by replacing n-gram chains (from <see cref="BucketTable"/>)
    /// with just the first token of each chain. This reduces effective sequence length before
    /// the forward pass during training-time sequence compression.
    /// </summary>
    private IReadOnlyList<int> CompressSequence(IReadOnlyList<int> tokenIds)
    {
        if (BucketTable is null || tokenIds.Count == 0)
        {
            return tokenIds;
        }

        var result = new List<int>(tokenIds.Count);
        var i = 0;
        while (i < tokenIds.Count)
        {
            // Use the prefix-indexed TryMatchAt for O(1) candidate lookup + O(chain_len) verification
            // instead of a linear scan over all buckets.
            if (BucketTable.TryMatchAt(tokenIds, i, out var bestMatch) && bestMatch is not null)
            {
                // Replace the matched n-gram with its first token only, shortening the sequence.
                result.Add(bestMatch.TokenIds[0]);
                i += bestMatch.TokenIds.Length;
            }
            else
            {
                result.Add(tokenIds[i]);
                i++;
            }
        }

        return result;
    }

    private IEnumerable<Layers.BitLinear> EnumerateBitLinearLayers()
    {
        foreach (var layer in EnumerateTransformerBitLinearLayers())
        {
            yield return layer;
        }

        yield return Transformer.OutputHead;
    }

    private IEnumerable<Layers.BitLinear> EnumerateTransformerBitLinearLayers()
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
    }

    private IEnumerable<Layers.RmsNorm> EnumerateNormLayers()
    {
        foreach (var layer in Transformer.Layers)
        {
            yield return layer.PreAttentionNorm;
            yield return layer.PreFeedForwardNorm;
        }

        yield return Transformer.FinalNorm;
    }

    private int GetId(string token) => _tokenToId.TryGetValue(token, out var id) ? id : _tokenToId[BitNetTokenizer.UnknownToken];
}
