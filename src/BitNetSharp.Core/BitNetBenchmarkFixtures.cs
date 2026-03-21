namespace BitNetSharp.Core;

public sealed record BitNetBenchmarkTextFixture(
    string Name,
    IReadOnlyList<string> Samples);

public static class BitNetBenchmarkFixtures
{
    private const string WikiText2ValidationResourceName = "BitNetSharp.Core.Data.WikiText2.wiki.valid.tokens";
    private static readonly Lazy<IReadOnlyList<string>> WikiText2ValidationSamplesLazy = new(LoadWikiText2ValidationSamples);

    public static IReadOnlyList<string> WikiText2ValidationSamples => WikiText2ValidationSamplesLazy.Value;

    public static IReadOnlyList<string> C4ValidationSamples { get; } =
    [
        "use the training command to review the loss chart",
        "microsoft agent framework hosting stays clear"
    ];

    public static IReadOnlyList<string> RedPajamaValidationSamples { get; } =
    [
        "visualize the ternary weight histogram",
        "how are you hosted with a local agent framework"
    ];

    public static IReadOnlyList<BitNetBenchmarkTextFixture> PerplexityDatasets { get; } =
    [
        new("WikiText2", WikiText2ValidationSamples),
        new("C4", C4ValidationSamples),
        new("RedPajama", RedPajamaValidationSamples)
    ];

    private static IReadOnlyList<string> LoadWikiText2ValidationSamples()
    {
        using var stream = typeof(BitNetBenchmarkFixtures).Assembly.GetManifestResourceStream(WikiText2ValidationResourceName)
            ?? throw new InvalidOperationException($"Could not load embedded resource '{WikiText2ValidationResourceName}'.");
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd()
            .Split('\n')
            .Select(static line => line.TrimEnd('\r'))
            .Where(static line => !string.IsNullOrWhiteSpace(line))
            .ToArray();
    }
}
