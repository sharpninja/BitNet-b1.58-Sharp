using BitNetSharp.Core;
using System.Text.Json;

namespace BitNetSharp.App;

public sealed record DataGenOptions(
    string Domain,
    int Count,
    string OutputPath,
    string TaskType,
    IReadOnlyList<string> Constraints,
    string? SeedPath,
    string OutputSchema,
    string? TemplatePath,
    string? LoraPath,
    string ModelSpecifier,
    VerbosityLevel Verbosity,
    int CandidateCount,
    double MinimumQualityScore,
    int? MaxOutputTokens)
{
    public const string DefaultTaskType = "instruction-response";
    public const string DefaultOutputSchema = """
        {"instruction":"string","response":"string","domain":"string","quality_score":"number","generation_timestamp":"ISO-8601"}
        """;

    public static DataGenOptions Parse(string[] args, string modelSpecifier, VerbosityLevel verbosity)
    {
        var domain = GetOption(args, "--domain=");
        if (string.IsNullOrWhiteSpace(domain))
        {
            throw new ArgumentException("The datagen command requires a non-empty --domain option.", nameof(args));
        }

        var outputPath = GetOption(args, "--output=");
        if (string.IsNullOrWhiteSpace(outputPath))
        {
            throw new ArgumentException("The datagen command requires a non-empty --output option.", nameof(args));
        }

        var count = ParsePositiveInt(GetOption(args, "--count="), defaultValue: 1, optionName: "--count");
        var candidateCount = ParsePositiveInt(GetOption(args, "--candidate-count="), defaultValue: 3, optionName: "--candidate-count");
        var minimumQualityScore = ParseDouble(GetOption(args, "--min-quality="), defaultValue: 0.45d, optionName: "--min-quality");
        if (minimumQualityScore is < 0d or > 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(args), "The --min-quality option must be between 0.0 and 1.0.");
        }

        var outputSchema = GetOption(args, "--output-schema=") ?? DefaultOutputSchema;
        var inlineConstraints = GetOptions(args, "--constraint=");
        var csvConstraints = SplitCsvOption(GetOption(args, "--constraints="));

        return new DataGenOptions(
            domain.Trim(),
            count,
            Path.GetFullPath(outputPath),
            GetOption(args, "--task-type=")?.Trim() ?? DefaultTaskType,
            inlineConstraints.Concat(csvConstraints).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
            GetOption(args, "--seeds="),
            outputSchema.Trim(),
            GetOption(args, "--template="),
            GetOption(args, "--lora="),
            modelSpecifier,
            verbosity,
            candidateCount,
            minimumQualityScore,
            ParseNullableInt(GetOption(args, "--max-tokens=")));
    }

    public string BuildSummary() =>
        JsonSerializer.Serialize(
            new
            {
                Domain,
                Count,
                OutputPath,
                TaskType,
                Constraints,
                SeedPath,
                OutputSchema,
                TemplatePath,
                LoraPath,
                ModelSpecifier,
                Verbosity,
                CandidateCount,
                MinimumQualityScore,
                MaxOutputTokens
            },
            new JsonSerializerOptions { WriteIndented = true });

    private static string? GetOption(IEnumerable<string> args, string prefix) =>
        args.FirstOrDefault(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            ?.Split('=', 2)
            .LastOrDefault();

    private static IReadOnlyList<string> GetOptions(IEnumerable<string> args, string prefix) =>
        args.Where(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .Select(argument => argument.Split('=', 2).Last())
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Select(value => value.Trim())
            .ToArray();

    private static IEnumerable<string> SplitCsvOption(string? value) =>
        string.IsNullOrWhiteSpace(value)
            ? []
            : value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static int ParsePositiveInt(string? value, int defaultValue, string optionName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (int.TryParse(value, out var parsed) && parsed > 0)
        {
            return parsed;
        }

        throw new ArgumentOutOfRangeException(nameof(value), $"The {optionName} option must be a positive integer.");
    }

    private static double ParseDouble(string? value, double defaultValue, string optionName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (double.TryParse(value, out var parsed))
        {
            return parsed;
        }

        throw new ArgumentException($"The {optionName} option must be a floating-point number.", nameof(value));
    }

    private static int? ParseNullableInt(string? value) =>
        int.TryParse(value, out var parsed) ? parsed : null;
}
