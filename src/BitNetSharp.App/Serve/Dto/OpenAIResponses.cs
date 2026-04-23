using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BitNetSharp.App.Serve.Dto;

public sealed record OpenAIModelEntry(
    string Id,
    string Object,
    long Created,
    [property: JsonPropertyName("owned_by")] string OwnedBy);

public sealed record OpenAIModelList(string Object, IReadOnlyList<OpenAIModelEntry> Data);

public sealed record OpenAIUsage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);

public sealed record OpenAIChoice(
    int Index,
    OpenAIChatMessage Message,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

public sealed record OpenAIChatCompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<OpenAIChoice> Choices,
    OpenAIUsage Usage);

public sealed record OpenAIChatDelta(string? Role, string? Content);

public sealed record OpenAIChunkChoice(
    int Index,
    OpenAIChatDelta Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

public sealed record OpenAIChatCompletionChunk(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<OpenAIChunkChoice> Choices);

public sealed record OpenAICompletionChoice(
    int Index,
    string Text,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

public sealed record OpenAICompletionResponse(
    string Id,
    string Object,
    long Created,
    string Model,
    IReadOnlyList<OpenAICompletionChoice> Choices,
    OpenAIUsage Usage);

public sealed record OpenAIErrorBody(
    string Message,
    string Type,
    string? Code,
    string? Param);

public sealed record OpenAIErrorEnvelope(OpenAIErrorBody Error);
