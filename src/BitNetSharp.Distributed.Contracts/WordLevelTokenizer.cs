using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace BitNetSharp.Distributed.Contracts;

/// <summary>
/// Minimal word-level tokenizer for the Truck Mate intent-
/// classification domain. Splits on whitespace + punctuation,
/// maps tokens to integer IDs via a vocabulary built from the
/// training corpus. Designed for narrow-domain SLMs where a
/// 2000-5000 token vocabulary covers the relevant surface and a
/// full BPE tokenizer adds unneeded complexity.
///
/// <para>
/// Special tokens:
///   0 = [PAD]  — padding for fixed-length batches
///   1 = [UNK]  — out-of-vocabulary fallback
///   2 = [BOS]  — beginning of sequence
///   3 = [EOS]  — end of sequence
///   4 = [USER] — marks the start of the user utterance
///   5 = [INTENT] — marks the start of the intent JSON
/// </para>
///
/// <para>
/// The tokenizer is deterministic and stateless after construction.
/// Vocabulary is serialized to/from a compact JSON file so it can
/// be shipped alongside the model weights and loaded on both the
/// coordinator (for pre-tokenization) and the worker (for
/// inference).
/// </para>
/// </summary>
public sealed class WordLevelTokenizer
{
    public const int PadId    = 0;
    public const int UnkId    = 1;
    public const int BosId    = 2;
    public const int EosId    = 3;
    public const int UserId   = 4;
    public const int IntentId = 5;
    public const int FirstUserTokenId = 6;

    private static readonly string[] SpecialTokens =
    {
        "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[USER]", "[INTENT]"
    };

    private static readonly Regex TokenSplitter = new(
        @"(\[USER\]|\[INTENT\]|[a-zA-Z0-9]+(?:'[a-zA-Z]+)?|[^\s])",
        RegexOptions.Compiled);

    private readonly Dictionary<string, int> _tokenToId;
    private readonly string[] _idToToken;

    /// <summary>Number of tokens in the vocabulary including specials.</summary>
    public int VocabSize => _idToToken.Length;

    private WordLevelTokenizer(Dictionary<string, int> tokenToId, string[] idToToken)
    {
        _tokenToId = tokenToId;
        _idToToken = idToToken;
    }

    /// <summary>
    /// Builds the vocabulary from a corpus of raw text lines. Keeps
    /// the top <paramref name="maxVocab"/> most-frequent tokens
    /// (after the 6 special tokens). Tokens that appear fewer than
    /// <paramref name="minFrequency"/> times are dropped.
    /// </summary>
    public static WordLevelTokenizer TrainFromCorpus(
        IEnumerable<string> lines,
        int maxVocab = 8000,
        int minFrequency = 2)
    {
        ArgumentNullException.ThrowIfNull(lines);
        if (maxVocab < FirstUserTokenId + 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxVocab));
        }

        var freq = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var line in lines)
        {
            foreach (var token in Tokenize(line))
            {
                // Skip the special markers during frequency counting.
                if (token is "[USER]" or "[INTENT]")
                {
                    continue;
                }

                freq.TryGetValue(token, out var count);
                freq[token] = count + 1;
            }
        }

        var topTokens = freq
            .Where(kvp => kvp.Value >= minFrequency)
            .OrderByDescending(kvp => kvp.Value)
            .ThenBy(kvp => kvp.Key, StringComparer.OrdinalIgnoreCase)
            .Take(maxVocab - FirstUserTokenId)
            .Select(kvp => kvp.Key)
            .ToList();

        var idToToken = new string[FirstUserTokenId + topTokens.Count];
        for (var i = 0; i < SpecialTokens.Length; i++)
        {
            idToToken[i] = SpecialTokens[i];
        }

        for (var i = 0; i < topTokens.Count; i++)
        {
            idToToken[FirstUserTokenId + i] = topTokens[i];
        }

        var tokenToId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < idToToken.Length; i++)
        {
            tokenToId[idToToken[i]] = i;
        }

        return new WordLevelTokenizer(tokenToId, idToToken);
    }

    /// <summary>
    /// Encodes a single text line into a sequence of integer token
    /// IDs. Wraps with [BOS] and [EOS]. Unknown tokens map to
    /// <see cref="UnkId"/>.
    /// </summary>
    public int[] Encode(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new[] { BosId, EosId };
        }

        var tokens = Tokenize(text);
        var ids = new List<int>(tokens.Count + 2) { BosId };

        foreach (var token in tokens)
        {
            if (_tokenToId.TryGetValue(token, out var id))
            {
                ids.Add(id);
            }
            else
            {
                ids.Add(UnkId);
            }
        }

        ids.Add(EosId);
        return ids.ToArray();
    }

    /// <summary>
    /// Decodes a sequence of token IDs back to text. Special tokens
    /// are rendered as their bracket-string forms.
    /// </summary>
    public string Decode(ReadOnlySpan<int> ids)
    {
        var sb = new StringBuilder();
        foreach (var id in ids)
        {
            if (id >= 0 && id < _idToToken.Length)
            {
                if (sb.Length > 0)
                {
                    sb.Append(' ');
                }

                sb.Append(_idToToken[id]);
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Saves the vocabulary as a JSON file the constructor can
    /// reload via <see cref="LoadFromFile"/>.
    /// </summary>
    public void SaveToFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = JsonSerializer.Serialize(_idToToken, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json, Encoding.UTF8);
    }

    /// <summary>
    /// Loads a vocabulary from a JSON file saved by
    /// <see cref="SaveToFile"/>.
    /// </summary>
    public static WordLevelTokenizer LoadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var json = File.ReadAllText(path, Encoding.UTF8);
        var idToToken = JsonSerializer.Deserialize<string[]>(json)
            ?? throw new InvalidOperationException($"Could not deserialize vocabulary from {path}.");

        var tokenToId = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < idToToken.Length; i++)
        {
            tokenToId[idToToken[i]] = i;
        }

        return new WordLevelTokenizer(tokenToId, idToToken);
    }

    /// <summary>
    /// Splits raw text into tokens using a regex that preserves the
    /// special [USER] and [INTENT] markers, splits on whitespace,
    /// and emits individual punctuation characters.
    /// </summary>
    private static List<string> Tokenize(string text)
    {
        var matches = TokenSplitter.Matches(text);
        var tokens = new List<string>(matches.Count);
        foreach (Match m in matches)
        {
            tokens.Add(m.Value);
        }

        return tokens;
    }
}
