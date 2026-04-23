using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BitNetSharp.App.Serve.Dto;

public sealed record OpenAIChatMessage(string Role, string Content);

public sealed record OpenAIChatCompletionRequest(
    string Model,
    IReadOnlyList<OpenAIChatMessage> Messages,
    [property: JsonPropertyName("max_tokens")] int? MaxTokens = null,
    double? Temperature = null,
    [property: JsonPropertyName("top_p")] double? TopP = null,
    bool? Stream = false,
    [property: JsonPropertyName("stream_options")] IReadOnlyDictionary<string, object?>? StreamOptions = null,
    string? User = null,
    [property: JsonPropertyName("response_format")] IReadOnlyDictionary<string, object?>? ResponseFormat = null);

public sealed record OpenAICompletionRequest(
    string Model,
    string Prompt,
    [property: JsonPropertyName("max_tokens")] int? MaxTokens = null,
    double? Temperature = null,
    [property: JsonPropertyName("top_p")] double? TopP = null,
    bool? Stream = false,
    string? User = null);
