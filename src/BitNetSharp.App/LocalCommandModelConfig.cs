using System.Text.Json;
using System.Text.Json.Serialization;

namespace BitNetSharp.App;

public enum LocalCommandPromptTransport
{
    StandardInput,
    FinalArgument
}

public sealed record LocalCommandModelConfig
{
    public string ModelId { get; init; } = string.Empty;

    public string DisplayName { get; init; } = string.Empty;

    public string ExecutablePath { get; init; } = string.Empty;

    public string[] Arguments { get; init; } = [];

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public LocalCommandPromptTransport PromptTransport { get; init; } = LocalCommandPromptTransport.StandardInput;

    public string PrimaryLanguage { get; init; } = "en-US";

    public string? WorkingDirectory { get; init; }

    public static LocalCommandModelConfig Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException($"The local command model configuration file '{fullPath}' does not exist.", fullPath);
        }

        var json = File.ReadAllText(fullPath);
        var config = JsonSerializer.Deserialize<LocalCommandModelConfig>(
            json,
            new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true,
                Converters = { new JsonStringEnumConverter() }
            });

        if (config is null)
        {
            throw new InvalidOperationException($"The local command model configuration file '{fullPath}' could not be parsed.");
        }

        if (string.IsNullOrWhiteSpace(config.ModelId))
        {
            throw new InvalidOperationException($"The local command model configuration file '{fullPath}' must define a non-empty modelId.");
        }

        if (string.IsNullOrWhiteSpace(config.DisplayName))
        {
            throw new InvalidOperationException($"The local command model configuration file '{fullPath}' must define a non-empty displayName.");
        }

        if (string.IsNullOrWhiteSpace(config.ExecutablePath))
        {
            throw new InvalidOperationException($"The local command model configuration file '{fullPath}' must define a non-empty executablePath.");
        }

        return config with
        {
            ExecutablePath = Path.GetFullPath(config.ExecutablePath),
            WorkingDirectory = string.IsNullOrWhiteSpace(config.WorkingDirectory)
                ? null
                : Path.GetFullPath(config.WorkingDirectory)
        };
    }
}
