using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BitNetSharp.App.Serve.Dto;

/// <summary>
/// Ollama request payloads. Field names are snake_case on the wire; the
/// shared <see cref="ServeJson"/> options enforce that. Records are
/// deserialization targets only; they do not allocate on the hot path.
/// </summary>
public sealed record OllamaChatMessage(
    string Role,
    string Content,
    [property: JsonPropertyName("images")] IReadOnlyList<string>? Images = null);

public sealed record OllamaChatRequest(
    string Model,
    IReadOnlyList<OllamaChatMessage> Messages,
    bool? Stream = true,
    [property: JsonPropertyName("keep_alive")] string? KeepAlive = null,
    IReadOnlyDictionary<string, object?>? Options = null,
    string? Format = null);

public sealed record OllamaGenerateRequest(
    string Model,
    string Prompt,
    bool? Stream = true,
    [property: JsonPropertyName("keep_alive")] string? KeepAlive = null,
    IReadOnlyDictionary<string, object?>? Options = null,
    string? Format = null,
    string? Suffix = null,
    string? System = null,
    string? Template = null,
    [property: JsonPropertyName("context")] IReadOnlyList<int>? Context = null,
    bool? Raw = null);

public sealed record OllamaShowRequest(string Model, bool Verbose = false);

public sealed record OllamaEmbeddingsRequest(
    string Model,
    string Prompt,
    [property: JsonPropertyName("keep_alive")] string? KeepAlive = null);
