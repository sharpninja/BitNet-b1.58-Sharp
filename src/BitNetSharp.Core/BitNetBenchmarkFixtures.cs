namespace BitNetSharp.Core;

public sealed record BitNetBenchmarkTextFixture(
    string Name,
    IReadOnlyList<string> Samples);

public static class BitNetBenchmarkFixtures
{
    public static IReadOnlyList<string> WikiText2ValidationSamples { get; } =
    [
        "hello i am bitnet sharp",
        "i default to american english"
    ];

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
}
