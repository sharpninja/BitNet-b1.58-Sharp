using System.Text.Json.Serialization;

namespace BitNetSharp.App;

public sealed record DataGenDatasetEntry(
    [property: JsonPropertyName("instruction")] string Instruction,
    [property: JsonPropertyName("response")] string Response,
    [property: JsonPropertyName("prompt")] string Prompt,
    [property: JsonPropertyName("domain")] string Domain,
    [property: JsonPropertyName("taskType")] string TaskType,
    [property: JsonPropertyName("qualityScore")] double QualityScore,
    [property: JsonPropertyName("generationTimestamp")] DateTimeOffset GenerationTimestamp,
    [property: JsonPropertyName("groundingContext")] IReadOnlyList<string> GroundingContext,
    [property: JsonPropertyName("lora")] string? LoraPath,
    [property: JsonPropertyName("seedInstruction")] string SeedInstruction,
    [property: JsonPropertyName("seedResponse")] string SeedResponse,
    [property: JsonPropertyName("variation")] string Variation,
    [property: JsonPropertyName("generatorModel")] string GeneratorModel,
    [property: JsonPropertyName("tags")] IReadOnlyList<string> Tags);
