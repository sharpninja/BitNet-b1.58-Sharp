using System.Text.Json;

namespace BitNetSharp.App;

public sealed record DataGenSeedExample(string Instruction, string Response)
{
    public static IReadOnlyList<DataGenSeedExample> LoadMany(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return [];
        }

        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"The DataGen seed file '{fullPath}' does not exist.", fullPath);
        }

        using var document = JsonDocument.Parse(File.ReadAllText(fullPath));
        if (document.RootElement.ValueKind != JsonValueKind.Array)
        {
            throw new InvalidOperationException($"The DataGen seed file '{fullPath}' must contain a JSON array.");
        }

        var seeds = new List<DataGenSeedExample>();
        foreach (var element in document.RootElement.EnumerateArray())
        {
            var instruction = ReadString(element, "instruction") ?? ReadString(element, "prompt");
            var response = ReadString(element, "response");

            if (string.IsNullOrWhiteSpace(instruction) || string.IsNullOrWhiteSpace(response))
            {
                throw new InvalidOperationException(
                    $"Each DataGen seed in '{fullPath}' must define either 'instruction' or 'prompt' plus a non-empty 'response'.");
            }

            seeds.Add(new DataGenSeedExample(instruction.Trim(), response.Trim()));
        }

        return seeds;
    }

    private static string? ReadString(JsonElement element, string propertyName) =>
        element.TryGetProperty(propertyName, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;
}
