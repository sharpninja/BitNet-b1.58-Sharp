using BitNetSharp.Core;
using System.Text.Json;

namespace BitNetSharp.App;

public sealed record DataGenCommandOptions(
    string Domain,
    int Count,
    string SeedsPath,
    string OutputPath,
    string? LoraPath)
{
    public static DataGenCommandOptions Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        var domain = ReadRequiredOption(args, "--domain");
        var seedsPath = ReadRequiredOption(args, "--seeds");
        var outputPath = ReadRequiredOption(args, "--output");
        var countValue = ReadRequiredOption(args, "--count");

        if (!int.TryParse(countValue, out var count) || count <= 0)
        {
            throw new ArgumentException("The datagen count must be a positive integer.", nameof(args));
        }

        return new DataGenCommandOptions(
            domain,
            count,
            Path.GetFullPath(seedsPath),
            Path.GetFullPath(outputPath),
            ReadOptionalOption(args, "--lora"));
    }

    private static string ReadRequiredOption(string[] args, string optionName)
    {
        var value = ReadOptionalOption(args, optionName);
        if (value is null)
        {
            throw new ArgumentException($"Missing required option '{optionName}'.", nameof(args));
        }

        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"Option '{optionName}' requires a non-empty value.", nameof(args));
        }

        return value;
    }

    private static string? ReadOptionalOption(string[] args, string optionName)
    {
        var equalsPrefix = $"{optionName}=";
        var requiresValue = false;
        for (var index = 0; index < args.Length; index++)
        {
            var argument = args[index];
            if (argument.StartsWith(equalsPrefix, StringComparison.OrdinalIgnoreCase))
            {
                return argument[equalsPrefix.Length..];
            }

            if (string.Equals(argument, optionName, StringComparison.OrdinalIgnoreCase))
            {
                var nextIndex = index + 1;
                if (nextIndex >= args.Length)
                {
                    requiresValue = true;
                    continue;
                }

                var nextArgument = args[nextIndex];
                if (nextArgument.StartsWith("--", StringComparison.Ordinal))
                {
                    requiresValue = true;
                    continue;
                }

                return nextArgument;
            }
        }

        if (requiresValue)
        {
            throw new ArgumentException($"Option '{optionName}' requires a value.", nameof(args));
        }

        return null;
    }
}

public static class DataGenCommand
{
    private static readonly JsonSerializerOptions OutputJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    public static async Task<string> RunAsync(string[] args, VerbosityLevel verbosity, CancellationToken cancellationToken = default)
    {
        var options = DataGenCommandOptions.Parse(args);
        var seeds = DataGenGenerator.LoadSeeds(options.SeedsPath);
        var generator = new DataGenGenerator(BitNetBootstrap.CreatePaperModel(verbosity));
        var outputDirectory = Path.GetDirectoryName(options.OutputPath);

        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        await using var stream = File.Create(options.OutputPath);
        await using var writer = new StreamWriter(stream);

        foreach (var example in generator.Generate(options.Domain, options.Count, seeds, options.LoraPath))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var line = JsonSerializer.Serialize(example, OutputJsonOptions);
            await writer.WriteLineAsync(line.AsMemory(), cancellationToken);
        }

        return options.OutputPath;
    }
}
