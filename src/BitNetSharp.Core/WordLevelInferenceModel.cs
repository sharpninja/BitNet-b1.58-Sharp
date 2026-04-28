using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using BitNetSharp.Core.Models;
using BitNetSharp.Distributed.Contracts;

namespace BitNetSharp.Core;

/// <summary>
/// Inference-only wrapper around a bare <see cref="BitNetTransformer"/> for
/// models trained via the distributed coordinator using
/// <see cref="WordLevelTokenizer"/>'s vocab layout (specials at fixed ids
/// <c>[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3, [USER]=4, [INTENT]=5</c>).
///
/// <para>
/// Mirrors <see cref="BitNetPaperModel"/>'s
/// <c>StreamGenerateAsync</c> API surface so the MAUI ChatPage and
/// IntentBenchPage can swap in a coordinator-trained checkpoint without
/// rewriting their consumer logic. Implementation is intentionally
/// minimal: greedy argmax sampling, optional repetition penalty, EOS
/// auto-stop, no chain-bucket speculative decoding. The narrow
/// intent-classification workload doesn't need the full BitNetPaperModel
/// scaffolding.
/// </para>
///
/// <para>
/// Streaming uses a producer task + Channel pattern identical to
/// BitNetPaperModel: caller awaits each <see cref="GeneratedToken"/>
/// without blocking the inference thread. Per-token timing fields
/// (<c>ForwardMs</c> / <c>SelectMs</c> / <c>DecodeMs</c>) are populated
/// the same way: <c>ForwardMs</c> = wall-clock of the forward that
/// produced this token; <c>DecodeMs</c> = wall-clock of the decode that
/// followed the previous token (zero on the first emitted token).
/// </para>
/// </summary>
public sealed class WordLevelInferenceModel
{
    private readonly BitNetTransformer _transformer;
    private readonly WordLevelTokenizer _tokenizer;

    public BitNetTransformer Transformer => _transformer;
    public WordLevelTokenizer Tokenizer => _tokenizer;

    /// <summary>Default cap for per-call generation when the caller
    /// passes no explicit <c>maxTokens</c>.</summary>
    public int MaxResponseTokens { get; init; } = 32;

    /// <summary>Repetition penalty applied to recent token ids before
    /// argmax. 1.0 disables the penalty. 1.3 is the same default
    /// BitNetPaperModel uses.</summary>
    public float RepetitionPenalty { get; init; } = 1.3f;

    /// <summary>How many recent ids the repetition penalty considers.</summary>
    public int RepetitionPenaltyWindow { get; init; } = 64;

    /// <summary>
    /// When true the sampler suppresses <see cref="WordLevelTokenizer.EosId"/>
    /// and <see cref="WordLevelTokenizer.UnkId"/> so the loop runs the full
    /// <c>maxTokens</c> regardless of model output. Useful for benchmarking
    /// per-token decode latency against under-trained checkpoints that
    /// argmax to EOS on every step.
    /// </summary>
    public bool SuppressEosAndUnk { get; init; }

    public WordLevelInferenceModel(BitNetTransformer transformer, WordLevelTokenizer tokenizer)
    {
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

        if (transformer.Config.VocabSize != tokenizer.VocabSize)
        {
            throw new ArgumentException(
                $"Transformer vocab size {transformer.Config.VocabSize} does not match tokenizer vocab size {tokenizer.VocabSize}.",
                nameof(tokenizer));
        }
    }

    public async IAsyncEnumerable<GeneratedToken> StreamGenerateAsync(
        string prompt,
        int? maxTokens = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);

        var promptIds = EncodePromptForGeneration(prompt);

        var maxOut = maxTokens ?? MaxResponseTokens;
        if (maxOut <= 0) yield break;

        var channel = Channel.CreateUnbounded<GeneratedToken>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
        });

        var producer = Task.Run(() =>
        {
            try
            {
                RunGenerationLoop(promptIds, maxOut, channel.Writer, cancellationToken);
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

    /// <summary>
    /// Synchronous one-shot generation that returns the full decoded
    /// string. Convenience wrapper for tests and the offline validator
    /// that don't care about streaming or per-token timing.
    /// </summary>
    public string GenerateResponse(string prompt, int? maxTokens = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(prompt);

        var promptIds = EncodePromptForGeneration(prompt);
        var maxOut = maxTokens ?? MaxResponseTokens;
        if (maxOut <= 0) return string.Empty;

        var emitted = new List<int>(maxOut);
        var capture = Channel.CreateUnbounded<GeneratedToken>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = true,
        });
        try
        {
            RunGenerationLoop(promptIds, maxOut, capture.Writer, cancellationToken);
        }
        finally
        {
            capture.Writer.TryComplete();
        }
        while (capture.Reader.TryRead(out var tok))
        {
            emitted.Add(tok.TokenId);
        }
        return DetokenizeIds(emitted);
    }

    private void RunGenerationLoop(
        int[] promptIds,
        int maxOut,
        ChannelWriter<GeneratedToken> writer,
        CancellationToken cancellationToken)
    {
        var capacity = _transformer.Config.MaxSequenceLength;
        // Cap prompt length: we need room for at least one generated token.
        if (promptIds.Length >= capacity)
        {
            // Truncate from the front, preserving BOS at index 0.
            var keep = capacity - 1;
            var trimmed = new int[keep];
            trimmed[0] = WordLevelTokenizer.BosId;
            Array.Copy(promptIds, promptIds.Length - keep + 1, trimmed, 1, keep - 1);
            promptIds = trimmed;
        }

        var cache = _transformer.CreateCache(capacity);

        var sw = Stopwatch.StartNew();

        // Prefill. Single forward over the entire prompt.
        var prefillStart = sw.Elapsed.TotalMilliseconds;
        var promptLogits = _transformer.Forward(promptIds, cache);
        var prefillMs = sw.Elapsed.TotalMilliseconds - prefillStart;

        // Initial select: argmax of the last prompt-position row.
        var generated = new List<int>(maxOut);
        var selectStart = sw.Elapsed.TotalMilliseconds;
        var nextId = SelectNextToken(promptLogits, promptLogits.GetLength(0) - 1, promptIds, generated);
        var selectMs = sw.Elapsed.TotalMilliseconds - selectStart;

        // Track ms attributed to the FORWARD that produced the token we
        // are about to emit, the SELECT that picked it, and the DECODE
        // that ran AFTER the previous emit. Prefill seeds ForwardMs for
        // step 0; DecodeMs is 0 for step 0.
        var emitForwardMs = prefillMs;
        var emitSelectMs = selectMs;
        var emitDecodeMs = 0d;

        for (var step = 0; step < maxOut; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var pieceText = ComputeStreamPiece(generated, nextId);
            writer.TryWrite(new GeneratedToken(
                nextId,
                pieceText,
                step,
                emitForwardMs,
                emitSelectMs,
                emitDecodeMs));

            generated.Add(nextId);

            if (!SuppressEosAndUnk && nextId == WordLevelTokenizer.EosId)
            {
                break;
            }

            if (cache.PastLength >= capacity)
            {
                // No room left in the cache; stop.
                break;
            }

            // Decode: single-token forward to advance the cache and get
            // logits for the next position.
            var decodeStart = sw.Elapsed.TotalMilliseconds;
            var stepLogits = _transformer.Forward(new[] { nextId }, cache);
            var decodeMs = sw.Elapsed.TotalMilliseconds - decodeStart;

            var nextSelectStart = sw.Elapsed.TotalMilliseconds;
            nextId = SelectNextToken(stepLogits, 0, promptIds, generated);
            var nextSelectMs = sw.Elapsed.TotalMilliseconds - nextSelectStart;

            emitForwardMs = decodeMs;
            emitSelectMs = nextSelectMs;
            emitDecodeMs = decodeMs;
        }
    }

    private int SelectNextToken(
        float[,] logits,
        int row,
        IReadOnlyList<int> promptIds,
        IReadOnlyList<int> generated)
    {
        var vocabSize = logits.GetLength(1);
        var penalized = new float[vocabSize];
        for (var v = 0; v < vocabSize; v++)
        {
            penalized[v] = logits[row, v];
        }

        if (RepetitionPenalty > 1.0001f)
        {
            // Apply penalty over the last RepetitionPenaltyWindow tokens
            // taken from generated only (don't penalise prompt tokens; the
            // model needs to be able to reference [USER] / [INTENT] etc.).
            var windowStart = Math.Max(0, generated.Count - RepetitionPenaltyWindow);
            for (var i = windowStart; i < generated.Count; i++)
            {
                var id = generated[i];
                if (id < 0 || id >= vocabSize) continue;
                if (penalized[id] > 0)
                {
                    penalized[id] /= RepetitionPenalty;
                }
                else
                {
                    penalized[id] *= RepetitionPenalty;
                }
            }
        }

        // Suppress tokens the model should never sample: PAD always, BOS
        // always (the model only ever sees BOS at position 0).
        penalized[WordLevelTokenizer.PadId] = float.NegativeInfinity;
        penalized[WordLevelTokenizer.BosId] = float.NegativeInfinity;
        if (SuppressEosAndUnk)
        {
            penalized[WordLevelTokenizer.EosId] = float.NegativeInfinity;
            penalized[WordLevelTokenizer.UnkId] = float.NegativeInfinity;
        }

        var bestId = WordLevelTokenizer.UnkId;
        var bestLogit = float.NegativeInfinity;
        for (var v = 0; v < vocabSize; v++)
        {
            if (penalized[v] > bestLogit)
            {
                bestLogit = penalized[v];
                bestId = v;
            }
        }
        return bestId;
    }

    private int[] EncodePromptForGeneration(string prompt) => _tokenizer.EncodeForGeneration(prompt);

    /// <summary>
    /// Computes the text piece that should be emitted for the new token,
    /// given the already-emitted ids. Mirrors BitNetTokenizer.Detokenize's
    /// "punctuation hugs the prior word" rule but uses WordLevelTokenizer's
    /// id->string mapping.
    /// </summary>
    private string ComputeStreamPiece(IReadOnlyList<int> alreadyEmitted, int newId)
    {
        var tokenStr = _tokenizer.GetTokenString(newId);
        if (alreadyEmitted.Count == 0)
        {
            return tokenStr;
        }
        if (IsAttachingToken(tokenStr))
        {
            return tokenStr;
        }
        return " " + tokenStr;
    }

    private static bool IsAttachingToken(string token)
    {
        // Single-char punctuation hugs the prior word. JSON-shaped tokens
        // ('{', '}', '[', ']', ',', ':', '"') all qualify under the
        // char.IsPunctuation rule plus the explicit closure-set fallback.
        if (token.Length == 1)
        {
            var c = token[0];
            return char.IsPunctuation(c) || c is '{' or '}' or '[' or ']' or ':' or '"';
        }
        return false;
    }

    /// <summary>
    /// Detokenizes a generated id sequence into a single readable string,
    /// skipping specials and applying the same attaching-punctuation rule
    /// as <see cref="ComputeStreamPiece"/>.
    /// </summary>
    public string DetokenizeIds(IReadOnlyList<int> ids)
    {
        var sb = new StringBuilder();
        var emitted = 0;
        foreach (var id in ids)
        {
            if (id < 0) continue;
            // Skip specials in the rendered string.
            if (id == WordLevelTokenizer.PadId || id == WordLevelTokenizer.BosId
                || id == WordLevelTokenizer.EosId || id == WordLevelTokenizer.UnkId)
            {
                continue;
            }
            var tokenStr = _tokenizer.GetTokenString(id);
            if (emitted == 0 || IsAttachingToken(tokenStr))
            {
                sb.Append(tokenStr);
            }
            else
            {
                sb.Append(' ');
                sb.Append(tokenStr);
            }
            emitted++;
        }
        return sb.ToString();
    }
}
