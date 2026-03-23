using System.Text.Json;
using BitNetSharp.Core;

namespace BitNetSharp.App;

internal sealed record TrainingDataset(
    string Name,
    IReadOnlyList<TrainingExample> Examples);

internal static class TrainingDatasetLoader
{
    public static TrainingDataset Load(string dataset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataset);

        return dataset.Trim().ToLowerInvariant() switch
        {
            "default" => new TrainingDataset("default", BitNetTrainingCorpus.CreateDefaultExamples()),
            "benchmark" or "tinyllama" or "tinyllama-1.1b" => new TrainingDataset(BitNetTrainingCorpus.BenchmarkDatasetName, BitNetTrainingCorpus.CreateBenchmarkExamples()),
            _ => LoadFromFile(dataset)
        };
    }

    private static TrainingDataset LoadFromFile(string path)
    {
        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"Training dataset file '{fullPath}' does not exist.", fullPath);
        }

        var examples = Path.GetExtension(fullPath).ToLowerInvariant() switch
        {
            ".jsonl" => LoadJsonLines(fullPath),
            ".json" => LoadJson(fullPath),
            _ => throw new ArgumentException($"Unsupported dataset format '{Path.GetExtension(fullPath)}'. Use .json or .jsonl.", nameof(path))
        };

        if (examples.Count == 0)
        {
            throw new InvalidOperationException($"The training dataset '{fullPath}' did not produce any prompt/response examples.");
        }

        return new TrainingDataset(Path.GetFileNameWithoutExtension(fullPath), examples);
    }

    private static IReadOnlyList<TrainingExample> LoadJson(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        return document.RootElement.ValueKind switch
        {
            JsonValueKind.Array => document.RootElement.EnumerateArray().Select(ParseExample).ToArray(),
            JsonValueKind.Object => [ParseExample(document.RootElement)],
            _ => throw new InvalidOperationException($"Dataset '{path}' must contain a JSON object or array.")
        };
    }

    private static IReadOnlyList<TrainingExample> LoadJsonLines(string path)
    {
        var examples = new List<TrainingExample>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var document = JsonDocument.Parse(line);
            examples.Add(ParseExample(document.RootElement));
        }

        return examples;
    }

    private static TrainingExample ParseExample(JsonElement element)
    {
        var prompt = GetString(element, "prompt", "question");
        var response = GetString(element, "response", "answer");
        if (!string.IsNullOrWhiteSpace(prompt) && !string.IsNullOrWhiteSpace(response))
        {
            return new TrainingExample(prompt!, response!);
        }

        var instruction = GetString(element, "instruction");
        var output = GetString(element, "output");
        if (!string.IsNullOrWhiteSpace(instruction) && !string.IsNullOrWhiteSpace(output))
        {
            var input = GetString(element, "input", "context");
            return new TrainingExample(
                string.IsNullOrWhiteSpace(input) ? instruction! : $"{instruction}\n\n{input}",
                output!);
        }

        if (GetArray(element, "messages", "conversations") is { } messages)
        {
            var pendingPrompt = default(string);
            foreach (var message in messages)
            {
                var role = GetString(message, "role", "from", "speaker");
                var content = GetString(message, "content", "value", "text");
                if (string.IsNullOrWhiteSpace(role) || string.IsNullOrWhiteSpace(content))
                {
                    continue;
                }

                var normalizedRole = role.Trim().ToLowerInvariant();
                if (normalizedRole is "user" or "human" or "prompt" or "instruction")
                {
                    pendingPrompt = content;
                    continue;
                }

                if (normalizedRole is "assistant" or "gpt" or "model" or "response" or "output"
                    && !string.IsNullOrWhiteSpace(pendingPrompt))
                {
                    return new TrainingExample(pendingPrompt!, content!);
                }
            }
        }

        throw new InvalidOperationException("Training datasets must expose prompt/response, instruction/output, or conversation-style records.");
    }

    private static string? GetString(JsonElement element, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String)
            {
                var value = property.GetString();
                if (!string.IsNullOrWhiteSpace(value))
                {
                    return value.Trim();
                }
            }
        }

        return null;
    }

    private static JsonElement[]? GetArray(JsonElement element, params string[] propertyNames)
    {
        foreach (var propertyName in propertyNames)
        {
            if (element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.Array)
            {
                return property.EnumerateArray().Select(static item => item.Clone()).ToArray();
            }
        }

        return null;
    }
}
