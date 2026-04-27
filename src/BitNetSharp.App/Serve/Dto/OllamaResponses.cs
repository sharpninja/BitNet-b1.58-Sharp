using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace BitNetSharp.App.Serve.Dto;

public sealed record OllamaVersionResponse(string Version);

public sealed record OllamaTagEntryDetails(
    [property: JsonPropertyName("parent_model")] string ParentModel,
    string Format,
    string Family,
    IReadOnlyList<string> Families,
    [property: JsonPropertyName("parameter_size")] string ParameterSize,
    [property: JsonPropertyName("quantization_level")] string QuantizationLevel);

public sealed record OllamaTagEntry(
    string Name,
    string Model,
    [property: JsonPropertyName("modified_at")] string ModifiedAt,
    long Size,
    string Digest,
    OllamaTagEntryDetails Details);

public sealed record OllamaTagListResponse(IReadOnlyList<OllamaTagEntry> Models);

public sealed record OllamaShowResponse(
    string Modelfile,
    string Parameters,
    string Template,
    OllamaTagEntryDetails Details,
    [property: JsonPropertyName("model_info")] IReadOnlyDictionary<string, object?> ModelInfo,
    IReadOnlyList<string> Capabilities);

/// <summary>
/// /api/chat streaming chunk. Intermediate chunks carry a content delta and
/// <c>done=false</c>; the terminal chunk carries <c>done=true</c> plus timing
/// metadata. Section A3 added optional per-token <c>forward_ms</c>,
/// <c>select_ms</c>, <c>decode_ms</c> fields on intermediate chunks so
/// streaming clients see per-token cost without parsing server logs. The
/// fields are nullable; the terminal chunk leaves them null and the
/// aggregate <c>eval_duration</c>/<c>prompt_eval_duration</c> fields stay
/// authoritative for summaries.
/// </summary>
public sealed record OllamaChatResponseChunk(
    string Model,
    [property: JsonPropertyName("created_at")] string CreatedAt,
    OllamaChatMessage Message,
    bool Done,
    [property: JsonPropertyName("done_reason")] string? DoneReason = null,
    [property: JsonPropertyName("total_duration")] long? TotalDuration = null,
    [property: JsonPropertyName("load_duration")] long? LoadDuration = null,
    [property: JsonPropertyName("prompt_eval_count")] int? PromptEvalCount = null,
    [property: JsonPropertyName("prompt_eval_duration")] long? PromptEvalDuration = null,
    [property: JsonPropertyName("eval_count")] int? EvalCount = null,
    [property: JsonPropertyName("eval_duration")] long? EvalDuration = null,
    [property: JsonPropertyName("forward_ms")] double? ForwardMs = null,
    [property: JsonPropertyName("select_ms")] double? SelectMs = null,
    [property: JsonPropertyName("decode_ms")] double? DecodeMs = null);

public sealed record OllamaGenerateResponseChunk(
    string Model,
    [property: JsonPropertyName("created_at")] string CreatedAt,
    string Response,
    bool Done,
    [property: JsonPropertyName("done_reason")] string? DoneReason = null,
    [property: JsonPropertyName("total_duration")] long? TotalDuration = null,
    [property: JsonPropertyName("load_duration")] long? LoadDuration = null,
    [property: JsonPropertyName("prompt_eval_count")] int? PromptEvalCount = null,
    [property: JsonPropertyName("prompt_eval_duration")] long? PromptEvalDuration = null,
    [property: JsonPropertyName("eval_count")] int? EvalCount = null,
    [property: JsonPropertyName("eval_duration")] long? EvalDuration = null,
    [property: JsonPropertyName("context")] IReadOnlyList<int>? Context = null);

public sealed record OllamaErrorResponse(string Error);
