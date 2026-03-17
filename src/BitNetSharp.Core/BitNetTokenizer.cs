using System.Text.RegularExpressions;

namespace BitNetSharp.Core;

public sealed partial class BitNetTokenizer
{
    public const string UnknownToken = "<unk>";
    public const string BeginToken = "<bos>";
    public const string EndToken = "<eos>";

    private readonly HashSet<string> _vocabulary;

    public BitNetTokenizer(IEnumerable<string> vocabulary)
    {
        _vocabulary = new HashSet<string>(vocabulary, StringComparer.Ordinal);
        _vocabulary.Add(UnknownToken);
        _vocabulary.Add(BeginToken);
        _vocabulary.Add(EndToken);
    }

    public IReadOnlyList<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [UnknownToken];
        }

        var tokens = TokenRegex()
            .Matches(text.ToLowerInvariant())
            .Select(match => Normalize(match.Value))
            .ToList();

        return tokens.Count == 0 ? [UnknownToken] : tokens;
    }

    public string Normalize(string token) => _vocabulary.Contains(token) ? token : UnknownToken;

    public string Detokenize(IEnumerable<string> tokens)
    {
        var builder = new List<string>();

        foreach (var token in tokens)
        {
            if (token is BeginToken or EndToken or UnknownToken)
            {
                continue;
            }

            if (builder.Count == 0 || IsPunctuation(token))
            {
                builder.Add(token);
                continue;
            }

            builder.Add($" {token}");
        }

        return string.Concat(builder)
            .Replace(" !", "!")
            .Replace(" ?", "?")
            .Replace(" .", ".")
            .Replace(" ,", ",")
            .Replace(" ;", ";")
            .Replace(" :", ":")
            .Trim();
    }

    private static bool IsPunctuation(string token) => token.Length == 1 && char.IsPunctuation(token[0]);

    [GeneratedRegex(@"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", RegexOptions.Compiled)]
    private static partial Regex TokenRegex();
}
