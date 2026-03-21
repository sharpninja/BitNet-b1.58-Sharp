namespace BitNetSharp.Core;

public sealed record BitNetBenchmarkTextFixture(
    string Name,
    IReadOnlyList<string> Samples);

public static class BitNetBenchmarkFixtures
{
    private const string BlankSeparatorLine = " ";
    private static readonly Lazy<IReadOnlyList<string>> WikiText2TrainingSamplesLazy = new(() => LoadWikiText2Split("wiki.train.tokens"));
    private static readonly Lazy<IReadOnlyList<string>> WikiText2ValidationSamplesLazy = new(() => LoadWikiText2Split("wiki.valid.tokens"));
    private static readonly Lazy<IReadOnlyList<string>> WikiText2TestSamplesLazy = new(() => LoadWikiText2Split("wiki.test.tokens"));

    public static IReadOnlyList<string> WikiText2TrainingSamples => WikiText2TrainingSamplesLazy.Value;
    public static IReadOnlyList<string> WikiText2ValidationSamples => WikiText2ValidationSamplesLazy.Value;
    public static IReadOnlyList<string> WikiText2TestSamples => WikiText2TestSamplesLazy.Value;

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

    private static IReadOnlyList<string> LoadWikiText2Split(string resourceFileName)
    {
        var resourceName = $"BitNetSharp.Core.Data.WikiText2.{resourceFileName}";
        using var stream = typeof(BitNetBenchmarkFixtures).Assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException($"Could not load internal WikiText-2 embedded resource '{resourceName}'. Verify the file is included as an EmbeddedResource in BitNetSharp.Core.csproj.");
        using var reader = new StreamReader(stream);
        var lines = new List<string>();

        while (reader.ReadLine() is { } line)
        {
            lines.Add(line);
        }

        return lines.AsReadOnly();
    }
}
