using System.Text.Json;

namespace BitNetSharp.Core;

public sealed record DataGenSeedExample(string Instruction, string Response);

public sealed record SyntheticDataExample(
    string Domain,
    string Instruction,
    string Response,
    string SeedInstruction,
    string SeedResponse,
    string Variation,
    string GeneratorModel,
    string? LoraAdapter,
    IReadOnlyList<string> Tags);

public sealed class DataGenGenerator(BitNetPaperModel model)
{
    private static readonly HashSet<string> FocusStopWords =
    [
        "a",
        "an",
        "and",
        "before",
        "for",
        "from",
        "how",
        "identify",
        "the",
        "this",
        "to",
        "with"
    ];

    private static readonly string[] InstructionPatterns =
    [
        "Create a {domain} task that starts from this seed: {seedInstruction}",
        "Write a grounded {domain} instruction that expands on: {seedInstruction}",
        "Turn this seed into a realistic {domain} request with clear constraints: {seedInstruction}",
        "Draft a practical {domain} prompt that keeps the original intent of: {seedInstruction}"
    ];

    private static readonly string[] ResponsePatterns =
    [
        "Use the seed response as the baseline: {seedResponse} Then adapt it for {domain} work with extra attention to {focus}.",
        "Start with: {seedResponse} Add a concise {domain} checklist that validates assumptions about {focus}.",
        "Answer in clear American English. Reuse the seed guidance, then tailor it to {domain} constraints and document {focus}.",
        "Build from the original answer: {seedResponse} Close with a short {domain} follow-up that covers {focus}."
    ];

    private readonly BitNetPaperModel _model = model ?? throw new ArgumentNullException(nameof(model));

    public IEnumerable<SyntheticDataExample> Generate(
        string domain,
        int count,
        IReadOnlyList<DataGenSeedExample> seeds,
        string? loraAdapter = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(domain);
        ArgumentNullException.ThrowIfNull(seeds);

        if (count <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), count, "Synthetic example count must be greater than zero.");
        }

        if (seeds.Count == 0)
        {
            throw new ArgumentException("At least one seed example is required.", nameof(seeds));
        }

        var normalizedDomain = domain.Trim();
        var seedContexts = seeds
            .Select(seed => new
            {
                Seed = seed,
                Focus = BuildFocus(seed, normalizedDomain)
            })
            .ToArray();

        for (var index = 0; index < count; index++)
        {
            var seedContext = seedContexts[index % seedContexts.Length];
            var variationIndex = index % InstructionPatterns.Length;
            var variation = $"pattern-{variationIndex + 1}";

            yield return new SyntheticDataExample(
                Domain: normalizedDomain,
                Instruction: FormatInstruction(seedContext.Seed, normalizedDomain, variationIndex, index),
                Response: FormatResponse(seedContext.Seed, normalizedDomain, variationIndex, seedContext.Focus, loraAdapter),
                SeedInstruction: seedContext.Seed.Instruction,
                SeedResponse: seedContext.Seed.Response,
                Variation: variation,
                GeneratorModel: _model.ModelId,
                LoraAdapter: string.IsNullOrWhiteSpace(loraAdapter) ? null : Path.GetFileName(loraAdapter),
                Tags: BuildTags(normalizedDomain, variation));
        }
    }

    public static IReadOnlyList<DataGenSeedExample> LoadSeeds(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var json = File.ReadAllText(path);
        var entries = JsonSerializer.Deserialize<List<DataGenSeedDocument>>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? [];

        if (entries.Count == 0)
        {
            throw new InvalidOperationException("Seed example files must contain at least one JSON object.");
        }

        return entries.Select(entry => entry.ToSeedExample()).ToArray();
    }

    private static string FormatInstruction(DataGenSeedExample seed, string domain, int variationIndex, int index)
    {
        var pattern = InstructionPatterns[variationIndex];
        return $"{pattern.Replace("{domain}", domain, StringComparison.Ordinal).Replace("{seedInstruction}", seed.Instruction, StringComparison.Ordinal)} [sample {index + 1}]";
    }

    private static string FormatResponse(DataGenSeedExample seed, string domain, int variationIndex, string focus, string? loraAdapter)
    {
        var response = ResponsePatterns[variationIndex]
            .Replace("{seedResponse}", seed.Response, StringComparison.Ordinal)
            .Replace("{domain}", domain, StringComparison.Ordinal)
            .Replace("{focus}", focus, StringComparison.Ordinal);

        if (string.IsNullOrWhiteSpace(loraAdapter))
        {
            return response;
        }

        return $"{response} Adapter hint: {Path.GetFileName(loraAdapter)}.";
    }

    private static IReadOnlyList<string> BuildTags(string domain, string variation) =>
    [
        "synthetic",
        "offline",
        variation,
        .. domain
            .Split([' ', '-', '_'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(token => token.ToLowerInvariant())
    ];

    private string BuildFocus(DataGenSeedExample seed, string domain)
    {
        var prompt = $"{domain} {seed.Instruction}";
        var result = _model.GenerateResponse(prompt, maxTokens: 3);
        var focusTerms = domain
            .Split([' ', '-', '_'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Concat(seed.Instruction.Split([' ', '-', '_', ',', '.', ':', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .Concat(result.Tokens)
            .Where(token => !string.IsNullOrWhiteSpace(token))
            .Select(token => token.Trim().ToLowerInvariant())
            .Where(token => token.Length >= 4)
            .Where(token => !FocusStopWords.Contains(token))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(3)
            .ToArray();

        return focusTerms.Length == 0
            ? "context, evidence, and next steps"
            : string.Join(", ", focusTerms);
    }

    private sealed record DataGenSeedDocument(
        string? Instruction,
        string? Prompt,
        string? Input,
        string? Response,
        string? Output,
        string? Answer)
    {
        public DataGenSeedExample ToSeedExample()
        {
            var instruction = FirstNonEmpty(Instruction, Prompt, Input);
            var response = FirstNonEmpty(Response, Output, Answer);

            if (instruction is null || response is null)
            {
                throw new InvalidOperationException("Each seed example must define an instruction/prompt/input field and a response/output/answer field.");
            }

            return new DataGenSeedExample(instruction, response);
        }

        private static string? FirstNonEmpty(params string?[] values) =>
            values.FirstOrDefault(value => !string.IsNullOrWhiteSpace(value))?.Trim();
    }
}
