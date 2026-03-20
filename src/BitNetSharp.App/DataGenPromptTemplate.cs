using BitNetSharp.Core;
using System.Text.Json;

namespace BitNetSharp.App;

public sealed record DataGenPromptTemplate(
    string Name,
    string SystemPrompt,
    string UserPrompt,
    string DefaultOutputSchema)
{
    public static DataGenPromptTemplate Load(string? path)
    {
        var resolvedPath = ResolvePath(path);
        if (!File.Exists(resolvedPath))
        {
            throw new FileNotFoundException($"The DataGen template '{resolvedPath}' does not exist.", resolvedPath);
        }

        var template = JsonSerializer.Deserialize<DataGenPromptTemplate>(
            File.ReadAllText(resolvedPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

        if (template is null)
        {
            throw new InvalidOperationException($"The DataGen template '{resolvedPath}' could not be parsed.");
        }

        if (string.IsNullOrWhiteSpace(template.Name) ||
            string.IsNullOrWhiteSpace(template.SystemPrompt) ||
            string.IsNullOrWhiteSpace(template.UserPrompt))
        {
            throw new InvalidOperationException($"The DataGen template '{resolvedPath}' must define name, systemPrompt, and userPrompt values.");
        }

        return template;
    }

    public string RenderPrompt(
        DataGenCommandOptions options,
        SyntheticDataExample example,
        int sampleNumber,
        IReadOnlyList<DataGenSeedExample> retrievedSeeds)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["domain"] = options.Domain,
            ["count"] = options.Count.ToString(),
            ["task_type"] = options.TaskType,
            ["seed_examples"] = FormatSeeds(retrievedSeeds),
            ["constraints"] = FormatConstraints(options.Constraints),
            ["output_schema"] = string.IsNullOrWhiteSpace(options.OutputSchema) ? DefaultOutputSchema : options.OutputSchema,
            ["sample_number"] = sampleNumber.ToString(),
            ["variation"] = example.Variation,
            ["seed_instruction"] = example.SeedInstruction,
            ["seed_response"] = example.SeedResponse
        };

        var systemPrompt = Expand(SystemPrompt, values);
        var userPrompt = Expand(UserPrompt, values);
        return $"{systemPrompt}{Environment.NewLine}{Environment.NewLine}{userPrompt}".Trim();
    }

    private static string ResolvePath(string? path)
    {
        if (!string.IsNullOrWhiteSpace(path))
        {
            return Path.GetFullPath(path);
        }

        return Path.Combine(AppContext.BaseDirectory, "templates", "datagen", "default.json");
    }

    private static string Expand(string template, IReadOnlyDictionary<string, string> values)
    {
        var expanded = template;
        foreach (var (key, value) in values)
        {
            expanded = expanded.Replace($"{{{key}}}", value, StringComparison.OrdinalIgnoreCase);
        }

        return expanded;
    }

    private static string FormatSeeds(IReadOnlyList<DataGenSeedExample> seeds) =>
        seeds.Count == 0
            ? "No seed examples were supplied."
            : string.Join(
                Environment.NewLine,
                seeds.Select((seed, index) => $"{index + 1}. Instruction: {seed.Instruction}{Environment.NewLine}   Response: {seed.Response}"));

    private static string FormatConstraints(IReadOnlyList<string> constraints) =>
        constraints.Count == 0
            ? "No additional constraints."
            : string.Join(Environment.NewLine, constraints.Select((constraint, index) => $"{index + 1}. {constraint}"));
}
