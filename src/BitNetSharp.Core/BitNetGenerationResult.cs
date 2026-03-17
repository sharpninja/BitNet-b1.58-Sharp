namespace BitNetSharp.Core;

public sealed record BitNetGenerationResult(
    string ResponseText,
    IReadOnlyList<string> Tokens,
    IReadOnlyList<string> Diagnostics);
