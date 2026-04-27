using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using BitNetSharp.Core.Bucketing;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Sampling;
using BitNetSharp.Core.Training;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Core;

public sealed class BitNetPaperModel
{
    private const int MaxPredictionLimit = 8;
    // Matches TraditionalLocalModel.MinimumProbability (1e-6f) so perplexity comparisons are
    // apples-to-apples. The previous 1e-9 floor penalized BitNet by ~6.9 nats per out-of-vocab
    // token vs the traditional baseline on the same WikiText2 slice, inflating reported ppl.
    private const double ProbabilityFloor = 1e-6d;

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
    private readonly ILogger<BitNetPaperModel> _logger;
    private readonly ILoggerFactory _loggerFactory;
    private BucketRecallHeatMap? _recallHeatMap;

    public BitNetPaperModel(
        IEnumerable<TrainingExample> trainingExamples,
        ILogger<BitNetPaperModel> logger,
        ILoggerFactory loggerFactory,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        BitNetConfig? config = null,
        int seed = 42)
        : this(
            new BitNetOptions(BitNetTrainingCorpus.CreateVocabulary(trainingExamples), verbosity),
            logger,
            loggerFactory,
            config,
            seed)
    {
    }

    public BitNetPaperModel(
        IEnumerable<TrainingExample> trainingExamples,
        ILogger<BitNetPaperModel> logger,
        ILoggerFactory loggerFactory,
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
                EnableSequenceCompression: enableSequenceCompression,
                UseIntegerForward: BitNetOptions.IntegerForwardEnvDefault),
            logger,
            loggerFactory,
            config,
            seed)
    {
    }

    public BitNetPaperModel(
        BitNetOptions options,
        ILogger<BitNetPaperModel> logger,
        ILoggerFactory loggerFactory,
        BitNetConfig? config = null,
        int seed = 42,
        IProgress<double>? constructionProgress = null,
        bool skipRandomInit = false)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(logger);
        ArgumentNullException.ThrowIfNull(loggerFactory);

        _logger = logger;
        _loggerFactory = loggerFactory;
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
        Transformer = new BitNetTransformer(Config, _loggerFactory.CreateLogger<BitNetTransformer>(), seed, constructionProgress, skipRandomInit);
    }

    public BitNetOptions Options { get; }

    public BitNetConfig Config { get; }

    public BitNetTransformer Transformer { get; }

    /// <summary>
    /// Optional chain-bucket table used for inference-time speculative decoding and
    /// training-time sequence compression. Populated via <see cref="LoadBucketTable"/>.
    /// </summary>
    public ChainBucketTable? BucketTable { get; private set; }

    /// <summary>
    /// Optional recall heat map that tracks per-token and per-chain accept/attempt counts
    /// during speculative decoding. Populated when a bucket table is loaded and
    /// <see cref="BitNetOptions.EnableRecallHeatMap"/> is set.
    /// </summary>
    public BucketRecallHeatMap? RecallHeatMap => _recallHeatMap;

    public string ModelId => "bitnet-b1.58-sharp";

    public BitNetTokenizer Tokenizer => _tokenizer;

    public long EstimateResidentParameterBytes() => Transformer.EstimateResidentParameterBytes();

    public string GetTokenString(int tokenId) => _idToToken[tokenId];

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

        if (Options.EnableRecallHeatMap)
        {
            _recallHeatMap = new BucketRecallHeatMap(Config.VocabSize);
        }
    }

    public static BitNetPaperModel CreateDefault(
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false,
        ILoggerFactory? loggerFactory = null)
    {
        var lf = loggerFactory ?? NullLoggerFactory.Instance;
        return PrimeDefaultExamples(new(
            new BitNetOptions(
                BitNetTrainingCorpus.CreateDefaultVocabulary(),
                verbosity,
                EnableChainBuckets: enableChainBuckets,
                EnableSequenceCompression: enableSequenceCompression,
                UseIntegerForward: BitNetOptions.IntegerForwardEnvDefault),
            lf.CreateLogger<BitNetPaperModel>(),
            lf));
    }

    public static BitNetPaperModel CreateForTrainingCorpus(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false,
        ILoggerFactory? loggerFactory = null)
    {
        var lf = loggerFactory ?? NullLoggerFactory.Instance;
        return new(
            trainingExamples,
            lf.CreateLogger<BitNetPaperModel>(),
            lf,
            verbosity,
            enableChainBuckets,
            enableSequenceCompression);
    }

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
        => GenerateResponse(prompt, maxTokens, emitToken: null, cancellationToken: default);

    public BitNetGenerationResult GenerateResponse(
        string prompt,
        int? maxTokens,
        Action<int>? emitToken,
        CancellationToken cancellationToken)
        => GenerateResponse(prompt, maxTokens, emitToken, onTokenEmitted: null, cancellationToken);

    /// <summary>
    /// Section A2 overload: same as the four-argument version but also fires
    /// a richer per-token callback that carries the decode timing for the
    /// emitted token. The rich event is fired on the iteration AFTER the
    /// token's decode finishes (so its <c>DecodeMs</c> field is populated),
    /// or at loop exit for the final token (with <c>DecodeMs = 0</c> because
    /// no follow-up decode runs).
    /// </summary>
    public BitNetGenerationResult GenerateResponse(
        string prompt,
        int? maxTokens,
        Action<int>? emitToken,
        Action<GeneratedToken>? onTokenEmitted,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var totalSw = System.Diagnostics.Stopwatch.StartNew();
        lock (_gate)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _logger.LogInformation(
                "GenerateResponse start prompt_chars={PromptChars} max_tokens={MaxTokens} default_max={DefaultMax}",
                prompt.Length,
                maxTokens,
                Options.MaxResponseTokens);

            _recallHeatMap?.ResetGenerationState();

            var diagnostics = new List<string>();
            var tokenizeSw = System.Diagnostics.Stopwatch.StartNew();
            var contextTokenIds = TokenizeToIds(prompt).ToList();
            tokenizeSw.Stop();
            _logger.LogDebug(
                "Tokenize prompt_tokens={PromptTokens} tokenize_ms={TokenizeMs:F1}",
                contextTokenIds.Count,
                tokenizeSw.Elapsed.TotalMilliseconds);

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
                _logger.LogInformation("Memorized response hit prompt_key_hash={PromptKeyHash}", promptKey.GetHashCode());
                var memorizedLimit = Math.Max(1, maxTokens.GetValueOrDefault(Options.MaxResponseTokens));
                var memorizedStep = 0;
                foreach (var tokenId in memorizedResponse)
                {
                    if (generatedTokenIds.Count >= memorizedLimit)
                    {
                        break;
                    }
                    if (tokenId == _endTokenId || tokenId == _tokenToId[BitNetTokenizer.UnknownToken])
                    {
                        continue;
                    }
                    cancellationToken.ThrowIfCancellationRequested();
                    generatedTokenIds.Add(tokenId);
                    emitToken?.Invoke(tokenId);
                    // KV-FU8: also fire the rich callback so StreamGenerateAsync
                    // produces events for memorized prompts. Memorized responses
                    // skip prefill/decode entirely; surface zero timings so
                    // streaming clients can distinguish memorized from
                    // autoregressive paths but still see the token stream.
                    onTokenEmitted?.Invoke(new GeneratedToken(
                        TokenId: tokenId,
                        TokenText: string.Empty,
                        Step: memorizedStep,
                        ForwardMs: 0d,
                        SelectMs: 0d,
                        DecodeMs: 0d));
                    memorizedStep++;
                }

                if (Options.Verbosity == VerbosityLevel.Verbose)
                {
                    diagnostics.Add("Resolved response from trained exemplar memory.");
                }
            }
            else
            {
                var maxGeneratedTokens = Math.Max(1, maxTokens.GetValueOrDefault(Options.MaxResponseTokens));
                _logger.LogInformation("Autoregressive loop start max_generated_tokens={MaxGeneratedTokens}", maxGeneratedTokens);
                var exitReason = "cap";

                var cache = Transformer.CreateCache(Config.MaxSequenceLength);
                var prefillSw = System.Diagnostics.Stopwatch.StartNew();
                var logits = Options.UseIntegerForward
                    ? Transformer.ForwardWithCacheInteger(contextTokenIds, cache)
                    : Transformer.Forward(contextTokenIds, cache);
                prefillSw.Stop();
                _logger.LogInformation(
                    "Prefill prompt_tokens={PromptTokens} prefill_ms={PrefillMs:F1}",
                    contextTokenIds.Count,
                    prefillSw.Elapsed.TotalMilliseconds);

                // A1: forward_ms reported per step is the timing of the forward
                // pass that produced the logits this step is sampling from.
                // Step 0 reads the prefill timing; step N+1 reads the prior
                // step's decode timing. Seeded with prefill so step 0 is
                // never zero.
                var lastForwardMs = prefillSw.Elapsed.TotalMilliseconds;

                // A2: deferred rich-event emission. pendingEvent holds the
                // GeneratedToken for the previous step with DecodeMs unfilled;
                // it fires at the top of the next iteration once DecodeMs is
                // known (= the decode just measured into lastForwardMs). The
                // final token flushes after the loop with DecodeMs = 0.
                GeneratedToken? pendingEvent = null;

                for (var step = 0; step < maxGeneratedTokens; step++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (pendingEvent.HasValue && onTokenEmitted is not null)
                    {
                        onTokenEmitted(pendingEvent.Value with { DecodeMs = lastForwardMs });
                        pendingEvent = null;
                    }
                    else if (pendingEvent.HasValue)
                    {
                        pendingEvent = null;
                    }

                    var stepSw = System.Diagnostics.Stopwatch.StartNew();
                    var nextToken = SelectNextToken(logits, contextTokenIds);
                    var selectMs = stepSw.Elapsed.TotalMilliseconds;
                    _logger.LogInformation(
                        "Step[{Step}] seq_len={SeqLen} forward_ms={ForwardMs:F1} select_ms={SelectMs:F1} token_id={TokenId} logit={Logit:F3}",
                        step,
                        contextTokenIds.Count,
                        lastForwardMs,
                        selectMs,
                        nextToken.TokenId,
                        nextToken.Logit);
                    if (nextToken.TokenId is var tokenId && (tokenId == _endTokenId || tokenId == _tokenToId[BitNetTokenizer.UnknownToken]))
                    {
                        exitReason = tokenId == _endTokenId ? "eos" : "unk";
                        _logger.LogInformation("Autoregressive loop exit reason={ExitReason} at step={Step}", exitReason, step);
                        break;
                    }

                    generatedTokenIds.Add(nextToken.TokenId);
                    contextTokenIds.Add(nextToken.TokenId);
                    emitToken?.Invoke(nextToken.TokenId);

                    if (onTokenEmitted is not null)
                    {
                        // TokenText is filled by the streaming wrapper - the
                        // synchronous loop does not detokenize. Step keeps the
                        // 0-based index. DecodeMs is finalized in the next
                        // iteration (or at flush below).
                        pendingEvent = new GeneratedToken(
                            nextToken.TokenId,
                            TokenText: string.Empty,
                            Step: step,
                            ForwardMs: lastForwardMs,
                            SelectMs: selectMs,
                            DecodeMs: 0d);
                    }

                    if (Options.Verbosity == VerbosityLevel.Verbose)
                    {
                        diagnostics.Add($"Prediction: token={_idToToken[nextToken.TokenId]}, logit={nextToken.Logit:0.###}");
                    }

                    if (contextTokenIds.Count >= Config.MaxSequenceLength)
                    {
                        exitReason = "context-full";
                        _logger.LogInformation("Autoregressive loop exit reason={ExitReason} at step={Step}", exitReason, step);
                        break;
                    }

                    // Decode step: feed just the newly selected token through
                    // the cache. The duration becomes next iteration's
                    // forward_ms (A1). The previously separate decode debug
                    // log is dropped because the step log line at the top of
                    // the next iteration carries the same number.
                    var decodeSw = System.Diagnostics.Stopwatch.StartNew();
                    logits = Options.UseIntegerForward
                        ? Transformer.ForwardWithCacheInteger(new[] { nextToken.TokenId }, cache)
                        : Transformer.Forward(new[] { nextToken.TokenId }, cache);
                    decodeSw.Stop();
                    lastForwardMs = decodeSw.Elapsed.TotalMilliseconds;

                    // Chain-bucket speculative decoding: after each normally generated token,
                    // check if the current context tail matches a known chain prefix.
                    // If so, speculatively accept chain tokens that the model also predicts,
                    // feeding each accepted token through the cache (one forward per accept).
                    if (Options.EnableChainBuckets && BucketTable is not null
                        && BucketTable.TryLookupPrefix(contextTokenIds, out var chain)
                        && chain is not null)
                    {
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

                        if (matchedPrefixLen > 0)
                        {
                            attemptedChains++;
                            _recallHeatMap?.RecordChainAttempt(chain.ChainId, chain.TokenIds, matchedPrefixLen);
                            var acceptedTokensForChain = 0;
                            for (var ci = matchedPrefixLen;
                                 ci < chain.TokenIds.Length
                                 && step < maxGeneratedTokens - 1
                                 && contextTokenIds.Count < Config.MaxSequenceLength;
                                 ci++)
                            {
                                var speculativeId = chain.TokenIds[ci];
                                if (speculativeId == _endTokenId || speculativeId == _tokenToId[BitNetTokenizer.UnknownToken])
                                {
                                    break;
                                }

                                attemptedChainTokens++;
                                var verifyToken = SelectNextToken(logits, contextTokenIds);
                                var verifyProbability = GetTargetProbability(logits, speculativeId);
                                if (verifyToken.TokenId != speculativeId || verifyProbability < Options.ChainBucketAcceptanceThreshold)
                                {
                                    break;
                                }

                                generatedTokenIds.Add(speculativeId);
                                contextTokenIds.Add(speculativeId);
                                emitToken?.Invoke(speculativeId);

                                step++;
                                acceptedTokensForChain++;
                                acceptedChainTokens++;
                                _recallHeatMap?.RecordTokenAccepted(chain.ChainId, speculativeId);

                                if (Options.Verbosity == VerbosityLevel.Verbose)
                                {
                                    diagnostics.Add(
                                        $"Speculation accepted: token={_idToToken[speculativeId]}, chain={chain.ChainId}, probability={verifyProbability:0.###}");
                                }

                                logits = Options.UseIntegerForward
                                    ? Transformer.ForwardWithCacheInteger(new[] { speculativeId }, cache)
                                    : Transformer.Forward(new[] { speculativeId }, cache);
                            }

                            if (acceptedTokensForChain > 0)
                            {
                                acceptedChains++;
                                _recallHeatMap?.RecordChainAccepted(chain.ChainId);
                            }
                        }
                    }
                }
                // A2: flush the final token's pending event with DecodeMs=0
                // because no follow-up decode runs (loop exited on EOS, UNK,
                // context-full, or hit cap).
                if (pendingEvent.HasValue && onTokenEmitted is not null)
                {
                    onTokenEmitted(pendingEvent.Value);
                }

                _logger.LogInformation(
                    "Autoregressive loop end exit_reason={ExitReason} generated_tokens={Generated} chain_attempts={ChainAttempts} chain_accepts={ChainAccepts}",
                    exitReason,
                    generatedTokenIds.Count,
                    attemptedChainTokens,
                    acceptedChainTokens);
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

            totalSw.Stop();
            _logger.LogInformation(
                "GenerateResponse end total_ms={TotalMs:F0} generated_tokens={GeneratedTokens} response_chars={ResponseChars}",
                totalSw.Elapsed.TotalMilliseconds,
                generatedTokens.Length,
                responseText.Length);
            return new BitNetGenerationResult(responseText, generatedTokens, diagnostics, chainBucketMetrics);
        }
    }

    /// <summary>
    /// Streams generated tokens one at a time. Producer runs the existing
    /// synchronous <see cref="GenerateResponse(string, int?, Action{int}?, CancellationToken)"/>
    /// loop on a worker task and writes each token to a channel; the caller
    /// consumes via <c>await foreach</c> and receives the detokenized piece.
    /// </summary>
    public async IAsyncEnumerable<GeneratedToken> StreamGenerateAsync(
        string prompt,
        int? maxTokens = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        var channel = Channel.CreateUnbounded<GeneratedToken>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true
        });
        var pendingPrefix = new List<int>();
        var producer = Task.Run(() =>
        {
            try
            {
                // A2: use the rich onTokenEmitted callback so per-token
                // ForwardMs/SelectMs/DecodeMs surface in the GeneratedToken.
                // The callback fires one iteration after token emission so
                // DecodeMs is populated for all but the final token.
                GenerateResponse(
                    prompt,
                    maxTokens,
                    emitToken: null,
                    onTokenEmitted: evt =>
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        pendingPrefix.Add(evt.TokenId);
                        var joined = _tokenizer.Detokenize(pendingPrefix.Select(id => _idToToken[id]));
                        var priorLength = evt.Step == 0
                            ? 0
                            : _tokenizer.Detokenize(pendingPrefix.Take(pendingPrefix.Count - 1).Select(id => _idToToken[id])).Length;
                        var piece = priorLength <= joined.Length ? joined[priorLength..] : string.Empty;
                        channel.Writer.TryWrite(evt with { TokenText = piece });
                    },
                    cancellationToken);
                channel.Writer.TryComplete();
            }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
        }, cancellationToken);

        await foreach (var token in channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
        {
            yield return token;
        }
        await producer.ConfigureAwait(false);
    }

    public double CalculatePerplexity(IEnumerable<string> validationSamples)
    {
        ArgumentNullException.ThrowIfNull(validationSamples);

        // Tokenize all samples up-front (parallel where possible — the tokenizer is stateless).
        var samples = validationSamples as IReadOnlyList<string> ?? validationSamples.ToArray();
        var tokenizedSamples = new IReadOnlyList<int>[samples.Count];
        if (samples.Count >= 4)
        {
            System.Threading.Tasks.Parallel.For(
                0,
                samples.Count,
                i => tokenizedSamples[i] = EncodeTokenIds(samples[i], appendEndToken: true));
        }
        else
        {
            for (var i = 0; i < samples.Count; i++)
            {
                tokenizedSamples[i] = EncodeTokenIds(samples[i], appendEndToken: true);
            }
        }

        var totalLoss = 0d;
        var totalTokens = 0;
        foreach (var tokenIds in tokenizedSamples)
        {
            if (tokenIds.Count < 2)
            {
                continue;
            }

            // Single forward pass per chunk of at most MaxSequenceLength tokens. Because the
            // attention is causal (position i only attends to positions 0..i), a single forward
            // pass on tokens[start..end] produces logits for every position at once, and row i
            // of the result predicts the token at position start + i + 1. This replaces the
            // previous O(L^3) "one forward pass per target token" approach with an O(L^2) one.
            var chunkStart = 0;
            while (chunkStart < tokenIds.Count - 1)
            {
                var chunkLength = Math.Min(Config.MaxSequenceLength, tokenIds.Count - chunkStart);
                var chunk = new int[chunkLength];
                for (var i = 0; i < chunkLength; i++)
                {
                    chunk[i] = tokenIds[chunkStart + i];
                }

                var logits = Transformer.Forward(chunk);
                var vocabSize = logits.GetLength(1);

                // Row i of logits predicts the token at chunk position i + 1 (i.e. absolute
                // position chunkStart + i + 1). We compute per-row negative log-likelihood for
                // all predictable positions in this chunk.
                for (var row = 0; row < chunkLength - 1; row++)
                {
                    var targetTokenId = tokenIds[chunkStart + row + 1];
                    totalLoss -= Math.Log(GetTargetProbabilityAtRow(logits, row, targetTokenId, vocabSize));
                    totalTokens++;
                }

                // Advance past the fully-predicted portion. If the chunk covered the final
                // token, we're done. Otherwise advance by (chunkLength - 1) so the next chunk's
                // first row predicts the next unseen target.
                if (chunkStart + chunkLength >= tokenIds.Count)
                {
                    break;
                }

                chunkStart += chunkLength - 1;
            }
        }

        return totalTokens == 0 ? 0d : Math.Exp(totalLoss / totalTokens);
    }

    private static double GetTargetProbabilityAtRow(float[,] logits, int row, int targetId, int vocabSize)
    {
        var maxLogit = double.NegativeInfinity;
        for (var column = 0; column < vocabSize; column++)
        {
            var value = logits[row, column];
            if (value > maxLogit)
            {
                maxLogit = value;
            }
        }

        var partition = 0d;
        var targetProbability = 0d;
        for (var column = 0; column < vocabSize; column++)
        {
            var probabilityMass = Math.Exp(logits[row, column] - maxLogit);
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

    public TernaryWeightStats GetTernaryWeightStats()
    {
        long negative = 0;
        long zero = 0;
        long positive = 0;

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

    /// <summary>
    /// Teacher-forced one-step forward for perplexity: predicts the token after position
    /// <paramref name="lastContextTokenIndex"/> using at most <see cref="BitNetConfig.MaxSequenceLength"/>
    /// context tokens (sliding window when the full prefix is longer).
    /// </summary>
    internal float[,] ForwardLogitsPerplexityStep(IReadOnlyList<int> tokenIds, int lastContextTokenIndex)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (lastContextTokenIndex < 0 || lastContextTokenIndex >= tokenIds.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(lastContextTokenIndex));
        }

        return ForwardLogits(SlicePerplexityContext(tokenIds, lastContextTokenIndex));
    }

    private IReadOnlyList<int> SlicePerplexityContext(IReadOnlyList<int> tokenIds, int lastIncludedIndex)
    {
        var length = lastIncludedIndex + 1;
        if (length > Config.MaxSequenceLength)
        {
            var skip = length - Config.MaxSequenceLength;
            return tokenIds.Skip(skip).Take(Config.MaxSequenceLength).ToArray();
        }

        return tokenIds.Take(length).ToArray();
    }

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

    /// <summary>
    /// Lazy enumerator over the 7 projection BitLinear layers per transformer block,
    /// in the canonical (q, k, v, out, gate, up, down) order. Enables streaming
    /// serializers to walk weights without materializing them all as FP32 first.
    /// </summary>
    internal IEnumerable<Layers.BitLinear> GetTransformerBitLinearLayers() => EnumerateTransformerBitLinearLayers();

    /// <summary>
    /// Lazy enumerator over RmsNorm layers in canonical order: per block
    /// (pre-attention, pre-feedforward) then final norm.
    /// </summary>
    internal IEnumerable<Layers.RmsNorm> GetNormLayers() => EnumerateNormLayers();

    /// <summary>
    /// Direct reference to the output-head BitLinear. Used by streaming
    /// serializers to avoid a ToFullPrecision allocation of the full weight
    /// matrix.
    /// </summary>
    internal Layers.BitLinear GetOutputHead() => Transformer.OutputHead;

    /// <summary>
    /// Returns the token-embedding matrix. Callers must treat the result as
    /// read-only.
    /// </summary>
    internal float[,] GetTokenEmbeddingsMatrix() => Transformer.ExportTokenEmbeddings();

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

    private (int TokenId, float Logit) SelectNextToken(float[,] logits, IReadOnlyList<int>? contextTokenIds = null)
    {
        var lastRow = logits.GetLength(0) - 1;
        var vocabSize = logits.GetLength(1);

        var penalty = Options.RepetitionPenalty;
        var window = Options.RepetitionPenaltyWindow;
        var penalized = ArrayPool<float>.Shared.Rent(vocabSize);
        try
        {
            for (var tokenId = 0; tokenId < vocabSize; tokenId++)
            {
                penalized[tokenId] = logits[lastRow, tokenId];
            }

            if (penalty != 1f && window > 0 && contextTokenIds is { Count: > 0 })
            {
                var windowStart = Math.Max(0, contextTokenIds.Count - window);
                var windowLength = contextTokenIds.Count - windowStart;
                var contextSlice = ArrayPool<int>.Shared.Rent(windowLength);
                try
                {
                    for (var i = 0; i < windowLength; i++)
                    {
                        contextSlice[i] = contextTokenIds[windowStart + i];
                    }

                    SamplingUtilities.ApplyRepetitionPenalty(
                        penalized.AsSpan(0, vocabSize),
                        contextSlice.AsSpan(0, windowLength),
                        penalty);
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(contextSlice);
                }
            }

            var selectedTokenId = _endTokenId;
            var selectedLogit = float.NegativeInfinity;
            for (var tokenId = 0; tokenId < vocabSize; tokenId++)
            {
                if (tokenId == _beginTokenId)
                {
                    continue;
                }

                var logit = penalized[tokenId];
                if (logit > selectedLogit)
                {
                    selectedTokenId = tokenId;
                    selectedLogit = logit;
                }
            }

            return (selectedTokenId, selectedLogit);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(penalized);
        }
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

/// <summary>
/// Single token emission from <see cref="BitNetPaperModel.StreamGenerateAsync"/>.
/// <para>
/// <c>TokenText</c> is the detokenized incremental slice (what the caller
/// should append).
/// </para>
/// <para>
/// Timing fields (added in Section A2 of the residual close-out): all in
/// milliseconds. <c>ForwardMs</c> is the duration of the forward pass that
/// produced the logits this token was sampled from (= prefill_ms for step 0,
/// = previous step's decode for step N+1). <c>SelectMs</c> is the
/// argmax/sample/repetition-penalty cost. <c>DecodeMs</c> is the duration of
/// the decode forward triggered after this token was emitted, which becomes
/// the next token's <c>ForwardMs</c>; it is 0 for the final emitted token
/// because no follow-up decode runs.
/// </para>
/// </summary>
public readonly record struct GeneratedToken(
    int TokenId,
    string TokenText,
    int Step,
    double ForwardMs,
    double SelectMs,
    double DecodeMs);
