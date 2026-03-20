using BitNetSharp.Core;
using System.Text.Json;

namespace BitNetSharp.App;

public sealed record DataGenCommandOptions(
    string Domain,
    int Count,
    string OutputPath,
    string TaskType,
    IReadOnlyList<string> Constraints,
    string? SeedsPath,
    string OutputSchema,
    string? TemplatePath,
    string? LoraPath,
    int CandidateCount,
    double MinimumQualityScore,
    int? MaxOutputTokens)
{
    public const string DefaultTaskType = "instruction-response";
    public const string DefaultOutputSchema = """
        {"instruction":"string","response":"string","domain":"string","taskType":"string","qualityScore":"number","generationTimestamp":"ISO-8601"}
        """;

    public static DataGenCommandOptions Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        var domain = ReadRequiredOption(args, "--domain");
        var outputPath = ReadRequiredOption(args, "--output");
        var countValue = ReadRequiredOption(args, "--count");

        if (!int.TryParse(countValue, out var count) || count <= 0)
        {
            throw new ArgumentException("The datagen count must be a positive integer.", nameof(args));
        }

        var candidateCount = ParsePositiveInt(ReadOptionalOption(args, "--candidate-count"), defaultValue: 3, optionName: "--candidate-count");
        var minimumQualityScore = ParseDouble(ReadOptionalOption(args, "--min-quality"), defaultValue: 0.45d, optionName: "--min-quality");
        if (minimumQualityScore is < 0d or > 1d)
        {
            throw new ArgumentOutOfRangeException(nameof(args), "The --min-quality option must be between 0.0 and 1.0.");
        }

        var inlineConstraints = ReadRepeatableOption(args, "--constraint");
        var csvConstraints = SplitCsvOption(ReadOptionalOption(args, "--constraints"));
        var seedsPath = ReadOptionalOption(args, "--seeds");

        return new DataGenCommandOptions(
            domain.Trim(),
            count,
            Path.GetFullPath(outputPath),
            ReadOptionalOption(args, "--task-type")?.Trim() ?? DefaultTaskType,
            inlineConstraints.Concat(csvConstraints).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
            string.IsNullOrWhiteSpace(seedsPath) ? null : Path.GetFullPath(seedsPath),
            ReadOptionalOption(args, "--output-schema")?.Trim() ?? DefaultOutputSchema,
            ReadOptionalOption(args, "--template"),
            ResolveOptionalPath(ReadOptionalOption(args, "--lora")),
            candidateCount,
            minimumQualityScore,
            ParseNullableInt(ReadOptionalOption(args, "--max-tokens")));
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
        var missingValueDetected = false;
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
                    missingValueDetected = true;
                    continue;
                }

                var nextArgument = args[nextIndex];
                if (nextArgument.StartsWith("--", StringComparison.Ordinal))
                {
                    missingValueDetected = true;
                    continue;
                }

                return nextArgument;
            }
        }

        if (missingValueDetected)
        {
            throw new ArgumentException($"Option '{optionName}' requires a value.", nameof(args));
        }

        return null;
    }

    private static IReadOnlyList<string> ReadRepeatableOption(string[] args, string optionName)
    {
        var values = new List<string>();
        var equalsPrefix = $"{optionName}=";

        for (var index = 0; index < args.Length; index++)
        {
            var argument = args[index];
            if (argument.StartsWith(equalsPrefix, StringComparison.OrdinalIgnoreCase))
            {
                var value = argument[equalsPrefix.Length..].Trim();
                if (!string.IsNullOrWhiteSpace(value))
                {
                    values.Add(value);
                }
                continue;
            }

            if (string.Equals(argument, optionName, StringComparison.OrdinalIgnoreCase) &&
                index + 1 < args.Length &&
                !args[index + 1].StartsWith("--", StringComparison.Ordinal))
            {
                var value = args[index + 1].Trim();
                if (!string.IsNullOrWhiteSpace(value))
                {
                    values.Add(value);
                }
            }
        }

        return values;
    }

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

    private static string? ResolveOptionalPath(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : Path.GetFullPath(value);
}

public static class DataGenCommand
{
    // Generate a 5x surplus of candidates so low-quality samples can be rejected with enough headroom
    // for typical small and medium runs without pushing runtime costs toward unbounded regeneration.
    private const int CandidateExpansionMultiplier = 5;

    // Schema validity matters most, consistency comes next, and diversity provides a smaller balancing signal
    // so repeated low-value phrasing does not dominate accepted output.
    private const double SchemaWeight = 0.4d;
    private const double ConsistencyWeight = 0.35d;
    private const double DiversityWeight = 0.25d;
    private const int DefaultQualityProbeMaxTokens = 16;

    // These delimiters intentionally keep tokenization lightweight and deterministic so diversity scoring
    // compares coarse lexical overlap without pulling in a heavier NLP dependency for JSONL generation.
    private static readonly char[] TokenDelimiters = [' ', '\r', '\n', '\t', ',', '.', ':', ';', '!', '?', '(', ')', '[', ']', '{', '}', '"', '\''];

    private static readonly JsonSerializerOptions OutputJsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    public static async Task<string> RunAsync(string[] args, VerbosityLevel verbosity, CancellationToken cancellationToken = default)
    {
        var options = DataGenCommandOptions.Parse(args);
        var template = DataGenPromptTemplate.Load(options.TemplatePath);
        var model = BitNetBootstrap.CreatePaperModel(verbosity);
        var seeds = LoadSeeds(options);
        var generator = new DataGenGenerator(model);
        var outputDirectory = Path.GetDirectoryName(options.OutputPath);

        if (!string.IsNullOrWhiteSpace(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }

        await using var stream = File.Create(options.OutputPath);
        await using var writer = new StreamWriter(stream);

        var acceptedEntries = new List<DataGenDatasetEntry>();
        var candidateTarget = options.Count * CandidateExpansionMultiplier;
        foreach (var example in generator.Generate(options.Domain, candidateTarget, seeds, options.LoraPath))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (acceptedEntries.Count >= options.Count)
            {
                break;
            }

            var prompt = template.RenderPrompt(options, example, acceptedEntries.Count + 1, seeds);
            var qualityScore = ComputeQualityScore(model, prompt, example.Response, acceptedEntries, options);
            if (qualityScore < options.MinimumQualityScore)
            {
                continue;
            }

            var entry = new DataGenDatasetEntry(
                example.Instruction,
                example.Response,
                prompt,
                example.Domain,
                options.TaskType,
                qualityScore,
                DateTimeOffset.UtcNow,
                seeds.Select(seed => seed.Instruction).ToArray(),
                options.LoraPath,
                example.SeedInstruction,
                example.SeedResponse,
                example.Variation,
                example.GeneratorModel,
                example.Tags);

            acceptedEntries.Add(entry);
            var line = JsonSerializer.Serialize(entry, OutputJsonOptions);
            await writer.WriteLineAsync(line.AsMemory(), cancellationToken);
        }

        if (acceptedEntries.Count < options.Count)
        {
            throw new InvalidOperationException(
                $"DataGen could only accept {acceptedEntries.Count} examples after evaluating {candidateTarget} candidates. Lower --min-quality, increase --candidate-count, or add seeds/constraints.");
        }

        return options.OutputPath;
    }

    private static IReadOnlyList<DataGenSeedExample> LoadSeeds(DataGenCommandOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.SeedsPath))
        {
            return DataGenGenerator.LoadSeeds(options.SeedsPath);
        }

        var fallbackInstruction = $"Create a {options.TaskType} example for the {options.Domain} domain.";
        var fallbackResponse = options.Constraints.Count == 0
            ? $"Write a grounded response for {options.Domain} in clear American English."
            : $"Write a grounded response for {options.Domain} that follows these constraints: {string.Join("; ", options.Constraints)}.";

        return [new DataGenSeedExample(fallbackInstruction, fallbackResponse)];
    }

    private static double ComputeQualityScore(
        BitNetPaperModel model,
        string prompt,
        string response,
        IReadOnlyList<DataGenDatasetEntry> acceptedEntries,
        DataGenCommandOptions options)
    {
        var candidates = Enumerable.Range(0, options.CandidateCount)
            .Select(_ => model.GenerateResponse(prompt, options.MaxOutputTokens ?? DefaultQualityProbeMaxTokens).ResponseText)
            .ToArray();

        var majorityCount = candidates
            .GroupBy(candidate => candidate, StringComparer.Ordinal)
            .Select(group => group.Count())
            .DefaultIfEmpty(0)
            .Max();

        var consistencyScore = majorityCount / (double)candidates.Length;
        var completenessScore = !string.IsNullOrWhiteSpace(prompt) && !string.IsNullOrWhiteSpace(response) ? 1d : 0d;
        var diversityScore = ComputeDiversityScore(response, acceptedEntries);
        return Math.Round((completenessScore * SchemaWeight) + (consistencyScore * ConsistencyWeight) + (diversityScore * DiversityWeight), 4);
    }

    private static double ComputeDiversityScore(string candidate, IReadOnlyList<DataGenDatasetEntry> acceptedEntries)
    {
        if (acceptedEntries.Count == 0)
        {
            return 1d;
        }

        var candidateTerms = Tokenize(candidate);
        var maxSimilarity = acceptedEntries
            .Select(entry => ComputeJaccardSimilarity(candidateTerms, Tokenize(entry.Response)))
            .DefaultIfEmpty(0d)
            .Max();

        return Math.Round(1d - maxSimilarity, 4);
    }

    private static double ComputeJaccardSimilarity(ISet<string> left, ISet<string> right)
    {
        if (left.Count == 0 && right.Count == 0)
        {
            return 1d;
        }

        var intersection = left.Intersect(right, StringComparer.Ordinal).Count();
        var union = left.Count + right.Count - intersection;
        return union == 0 ? 0d : intersection / (double)union;
    }

    private static HashSet<string> Tokenize(string value) =>
        value.Split(TokenDelimiters,
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(term => term.ToLowerInvariant())
            .ToHashSet(StringComparer.Ordinal);
}
