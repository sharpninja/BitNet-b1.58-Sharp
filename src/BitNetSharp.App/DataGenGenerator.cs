using System.Text;
using System.Text.Json;

namespace BitNetSharp.App;

public sealed class DataGenGenerator(IHostedAgentModel model, DataGenPromptTemplate template)
{
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };

    public async Task<IReadOnlyList<DataGenDatasetEntry>> GenerateAsync(
        DataGenOptions options,
        CancellationToken cancellationToken = default)
    {
        var seeds = DataGenSeedExample.LoadMany(options.SeedPath);
        var accepted = new List<DataGenDatasetEntry>();
        var attempts = 0;
        var maxAttempts = options.Count * 10;

        while (accepted.Count < options.Count && attempts < maxAttempts)
        {
            attempts++;
            var sampleNumber = accepted.Count + 1;
            var retrievedSeeds = RetrieveRelevantSeeds(seeds, options, sampleNumber);
            var prompt = template.RenderPrompt(options, sampleNumber, retrievedSeeds);
            var instruction = BuildInstruction(options, retrievedSeeds, sampleNumber);
            var candidateResponses = await GenerateCandidatesAsync(prompt, options, retrievedSeeds, sampleNumber, cancellationToken);
            var selectedResponse = SelectMajorityResponse(candidateResponses, out var consistencyScore);
            var diversityScore = ComputeDiversityScore(selectedResponse, accepted);
            var schemaScore = ValidateRecord(instruction, selectedResponse, prompt, options) ? 1d : 0d;
            var qualityScore = Math.Round((schemaScore * 0.4d) + (consistencyScore * 0.35d) + (diversityScore * 0.25d), 4);

            if (qualityScore < options.MinimumQualityScore)
            {
                continue;
            }

            accepted.Add(
                new DataGenDatasetEntry(
                    instruction,
                    selectedResponse,
                    prompt,
                    options.Domain,
                    options.TaskType,
                    qualityScore,
                    DateTimeOffset.UtcNow,
                    retrievedSeeds.Select(seed => seed.Instruction).ToArray(),
                    options.LoraPath is null ? null : Path.GetFullPath(options.LoraPath)));
        }

        if (accepted.Count < options.Count)
        {
            throw new InvalidOperationException(
                $"DataGen could only accept {accepted.Count} examples after {attempts} attempts. Lower --min-quality or add more seeds/constraints.");
        }

        return accepted;
    }

    public static void WriteJsonl(string outputPath, IReadOnlyList<DataGenDatasetEntry> dataset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        var fullPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllLines(fullPath, dataset.Select(entry => JsonSerializer.Serialize(entry, JsonOptions)));
    }

    private async Task<IReadOnlyList<string>> GenerateCandidatesAsync(
        string prompt,
        DataGenOptions options,
        IReadOnlyList<DataGenSeedExample> retrievedSeeds,
        int sampleNumber,
        CancellationToken cancellationToken)
    {
        var candidates = new List<string>(options.CandidateCount);
        for (var candidateIndex = 0; candidateIndex < options.CandidateCount; candidateIndex++)
        {
            var result = await model.GetResponseAsync(prompt, options.MaxOutputTokens, cancellationToken);
            var groundedResponse = BuildGroundedResponse(result.Text, options, retrievedSeeds, sampleNumber, candidateIndex);
            candidates.Add(groundedResponse);
        }

        return candidates;
    }

    private static IReadOnlyList<DataGenSeedExample> RetrieveRelevantSeeds(
        IReadOnlyList<DataGenSeedExample> seeds,
        DataGenOptions options,
        int sampleNumber)
    {
        if (seeds.Count == 0)
        {
            return [];
        }

        var queryTerms = Tokenize($"{options.Domain} {options.TaskType} {string.Join(' ', options.Constraints)}");
        return seeds
            .Select((seed, index) => new
            {
                Seed = seed,
                Index = index,
                Score = ScoreSeed(seed, queryTerms, sampleNumber)
            })
            .OrderByDescending(item => item.Score)
            .ThenBy(item => (item.Index + sampleNumber) % Math.Max(seeds.Count, 1))
            .Take(Math.Min(3, seeds.Count))
            .Select(item => item.Seed)
            .ToArray();
    }

    private static int ScoreSeed(DataGenSeedExample seed, ISet<string> queryTerms, int sampleNumber)
    {
        var seedTerms = Tokenize($"{seed.Instruction} {seed.Response}");
        var overlap = seedTerms.Count(term => queryTerms.Contains(term));
        return overlap + ((seedTerms.Count + sampleNumber) % 3);
    }

    private static string BuildInstruction(
        DataGenOptions options,
        IReadOnlyList<DataGenSeedExample> retrievedSeeds,
        int sampleNumber)
    {
        var builder = new StringBuilder();
        builder.Append($"Create a {options.TaskType} training example for the {options.Domain} domain");
        builder.Append($" (sample {sampleNumber} of {options.Count}).");

        if (options.Constraints.Count > 0)
        {
            builder.Append(' ');
            builder.Append("Follow these constraints: ");
            builder.Append(string.Join("; ", options.Constraints));
            builder.Append('.');
        }

        if (retrievedSeeds.Count > 0)
        {
            builder.Append(' ');
            builder.Append("Ground the answer using themes from: ");
            builder.Append(string.Join("; ", retrievedSeeds.Select(seed => seed.Instruction)));
            builder.Append('.');
        }

        return builder.ToString();
    }

    private static string SelectMajorityResponse(IReadOnlyList<string> candidates, out double consistencyScore)
    {
        var majority = candidates
            .GroupBy(candidate => candidate, StringComparer.Ordinal)
            .OrderByDescending(group => group.Count())
            .ThenBy(group => group.Key, StringComparer.Ordinal)
            .First();

        consistencyScore = majority.Count() / (double)candidates.Count;
        return majority.Key;
    }

    private static double ComputeDiversityScore(string candidate, IReadOnlyList<DataGenDatasetEntry> accepted)
    {
        if (accepted.Count == 0)
        {
            return 1d;
        }

        var candidateTerms = Tokenize(candidate);
        var maxSimilarity = accepted
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
        var union = left.Union(right, StringComparer.Ordinal).Count();
        return union == 0 ? 0d : intersection / (double)union;
    }

    private static HashSet<string> Tokenize(string value) =>
        value.Split([' ', '\r', '\n', '\t', ',', '.', ':', ';', '!', '?', '(', ')', '[', ']', '{', '}', '"', '\''],
                StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(term => term.ToLowerInvariant())
            .ToHashSet(StringComparer.Ordinal);

    private static bool ValidateRecord(string instruction, string response, string prompt, DataGenOptions options) =>
        !string.IsNullOrWhiteSpace(instruction) &&
        !string.IsNullOrWhiteSpace(response) &&
        !string.IsNullOrWhiteSpace(prompt) &&
        !string.IsNullOrWhiteSpace(options.Domain) &&
        response.Contains(options.Domain, StringComparison.OrdinalIgnoreCase);

    private static string BuildGroundedResponse(
        string modelResponse,
        DataGenOptions options,
        IReadOnlyList<DataGenSeedExample> retrievedSeeds,
        int sampleNumber,
        int candidateIndex)
    {
        var grounding = retrievedSeeds.Count == 0
            ? "No retrieved seed context."
            : string.Join(" | ", retrievedSeeds.Select(seed => $"{seed.Instruction} => {seed.Response}"));

        return
            $"Domain: {options.Domain}{Environment.NewLine}" +
            $"Task type: {options.TaskType}{Environment.NewLine}" +
            $"Sample: {sampleNumber}{Environment.NewLine}" +
            $"Candidate: {candidateIndex + 1}{Environment.NewLine}" +
            $"Constraints: {(options.Constraints.Count == 0 ? "None" : string.Join("; ", options.Constraints))}{Environment.NewLine}" +
            $"Grounding: {grounding}{Environment.NewLine}" +
            $"Model synthesis: {modelResponse.Trim()}";
    }
}
