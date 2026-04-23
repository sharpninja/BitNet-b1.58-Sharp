using System.Globalization;
using BitNetSharp.Core;
using BitNetSharp.Core.Converters;

namespace BitNetSharp.App;

/// <summary>
/// `import-gguf` subcommand. Converts a Qwen3-architecture Prism-Q2_0 GGUF
/// (e.g. prism-ml/Ternary-Bonsai-8B-gguf) into a BitNetSharp-native GGUF by
/// collapsing quaternary codes to ternary trits and writing the result via
/// <see cref="BitNetPaperGguf.Save"/>.
/// </summary>
public static class ImportGgufCommand
{
    public static int Run(string[] args, VerbosityLevel verbosity)
    {
        ArgumentNullException.ThrowIfNull(args);

        var inputPath = ParseOption(args, "--input=");
        var outputPath = ParseOption(args, "--output=");
        var seedRaw = ParseOption(args, "--seed=");

        if (string.IsNullOrWhiteSpace(inputPath))
        {
            throw new ArgumentException("The import-gguf command requires --input=<source.gguf>.");
        }
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("The import-gguf command requires --output=<target.gguf>.");
        }
        if (!File.Exists(inputPath))
        {
            throw new FileNotFoundException($"Source GGUF not found: {inputPath}", inputPath);
        }

        int seed = 42;
        if (!string.IsNullOrWhiteSpace(seedRaw)
            && int.TryParse(seedRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedSeed))
        {
            seed = parsedSeed;
        }

        var options = new BitNetOptions(
            Vocabulary: BitNetTrainingCorpus.CreateDefaultVocabulary(),
            Verbosity: verbosity);

        if (verbosity != VerbosityLevel.Quiet)
        {
            Console.WriteLine($"Importing Qwen3/Prism-Q2_0 GGUF: {Path.GetFullPath(inputPath)}");
            Console.WriteLine("(body weights collapse quaternary -> ternary; token_embd + output are discarded and reseeded)");
        }

        var model = Qwen3BonsaiConverter.Convert(inputPath, options, seed);

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)) ?? ".");
        BitNetPaperGguf.Save(model, outputPath);

        if (verbosity != VerbosityLevel.Quiet)
        {
            Console.WriteLine($"Saved BitNetSharp GGUF to {Path.GetFullPath(outputPath)}");
            Console.WriteLine($"Layers: {model.Config.LayerCount}, Dim: {model.Config.Dimension}, "
                + $"Heads: {model.Config.HeadCount}Q/{model.Config.KvHeadCount}KV, "
                + $"Hidden: {model.Config.HiddenDimension}, Vocab: {model.Config.VocabSize}.");
        }

        return 0;
    }

    private static string? ParseOption(IEnumerable<string> args, string prefix) =>
        args.FirstOrDefault(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            ?.Split('=', 2)
            .LastOrDefault();
}
