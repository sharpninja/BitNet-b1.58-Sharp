using System.Text.Json.Serialization;

namespace BitNetSharp.App;

public sealed record DataGenDatasetEntry(
    [property: JsonPropertyName("instruction")] string Instruction,
    [property: JsonPropertyName("response")] string Response,
    [property: JsonPropertyName("prompt")] string Prompt,
    [property: JsonPropertyName("domain")] string Domain,
    [property: JsonPropertyName("task_type")] string TaskType,
    [property: JsonPropertyName("quality_score")] double QualityScore,
    [property: JsonPropertyName("generation_timestamp")] DateTimeOffset GenerationTimestamp,
    [property: JsonPropertyName("grounding_context")] IReadOnlyList<string> GroundingContext,
    [property: JsonPropertyName("lora")] string? LoraPath);
