namespace BitNetSharp.Core;

public sealed record BitNetOptions(
    IReadOnlyList<string> Vocabulary,
    VerbosityLevel Verbosity = VerbosityLevel.Normal,
    int MaxResponseTokens = 24,
    string PrimaryLanguage = "en-US");
