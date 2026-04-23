using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using BitNetSharp.App.Serve.Dto;

namespace BitNetSharp.App.Serve;

/// <summary>
/// Descriptive snapshot of a loaded model, computed once at registry build
/// time. The <see cref="Name"/> carries Ollama's mandatory <c>:latest</c>
/// tag suffix because Open WebUI's model picker splits on the colon.
/// </summary>
public sealed record ModelCard(
    string Name,
    string NameWithTag,
    long ParameterCount,
    long SizeBytes,
    string ModifiedAt,
    string Digest,
    OllamaTagEntryDetails Details,
    IReadOnlyDictionary<string, object?> ModelInfo)
{
    public static ModelCard ForHostedModel(IHostedAgentModel model, long parameterCount, long sizeBytes, IReadOnlyDictionary<string, object?> modelInfo)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(modelInfo);

        string baseName = model.ModelId;
        string nameWithTag = baseName.Contains(':', StringComparison.Ordinal) ? baseName : baseName + ":latest";
        string modifiedAt = DateTimeOffset.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", System.Globalization.CultureInfo.InvariantCulture);
        string digest = ComputeDigest(baseName, parameterCount, sizeBytes);

        var details = new OllamaTagEntryDetails(
            ParentModel: string.Empty,
            Format: "gguf",
            Family: "bitnet",
            Families: new[] { "bitnet" },
            ParameterSize: FormatParameterSize(parameterCount),
            QuantizationLevel: "b1.58");

        return new ModelCard(baseName, nameWithTag, parameterCount, sizeBytes, modifiedAt, digest, details, modelInfo);
    }

    public OllamaTagEntry ToTagEntry() =>
        new(
            Name: NameWithTag,
            Model: NameWithTag,
            ModifiedAt: ModifiedAt,
            Size: SizeBytes,
            Digest: Digest,
            Details: Details);

    public OpenAIModelEntry ToOpenAIEntry()
    {
        long created = DateTimeOffset.Parse(ModifiedAt, System.Globalization.CultureInfo.InvariantCulture).ToUnixTimeSeconds();
        return new OpenAIModelEntry(Id: Name, Object: "model", Created: created, OwnedBy: "bitnetsharp");
    }

    private static string ComputeDigest(string modelId, long parameterCount, long sizeBytes)
    {
        string seed = $"{modelId}|{parameterCount}|{sizeBytes}";
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(seed));
        var sb = new StringBuilder("sha256:", 71);
        foreach (byte b in hash)
        {
            sb.Append(b.ToString("x2", System.Globalization.CultureInfo.InvariantCulture));
        }
        return sb.ToString();
    }

    private static string FormatParameterSize(long parameterCount)
    {
        if (parameterCount <= 0) return "unknown";
        double billions = parameterCount / 1_000_000_000d;
        if (billions >= 1d)
        {
            return $"{billions:0.#}B";
        }
        double millions = parameterCount / 1_000_000d;
        if (millions >= 1d)
        {
            return $"{millions:0.#}M";
        }
        return $"{parameterCount}";
    }
}
