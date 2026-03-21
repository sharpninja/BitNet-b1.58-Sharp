using BitNetSharp.Core;

namespace BitNetSharp.Tests;

internal static class BenchmarkFixtureTestData
{
    private const string BlankSeparatorLine = " ";

    public static IReadOnlyList<string> CreateCompactWikiText2ValidationSamples() =>
        BitNetBenchmarkFixtures.WikiText2ValidationSamples
            .Where(static sample => string.Equals(sample, BlankSeparatorLine, StringComparison.Ordinal)
                || sample.StartsWith(" = ", StringComparison.Ordinal))
            .Take(8)
            .ToArray();

    public static IReadOnlyList<BitNetBenchmarkTextFixture> CreateCompactPerplexityDatasets() =>
    [
        new("WikiText2", CreateCompactWikiText2ValidationSamples()),
        new("C4", BitNetBenchmarkFixtures.C4ValidationSamples),
        new("RedPajama", BitNetBenchmarkFixtures.RedPajamaValidationSamples)
    ];
}
