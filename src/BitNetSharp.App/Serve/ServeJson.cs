using System.Text.Json;

namespace BitNetSharp.App.Serve;

/// <summary>
/// Shared JSON options for the serve surface. snake_case because both Ollama
/// and OpenAI use it on their wire formats. DefaultIgnoreCondition drops null
/// properties so terminal response chunks don't carry stale timing fields and
/// error envelopes stay minimal on the wire.
/// </summary>
internal static class ServeJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DictionaryKeyPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = false,
    };
}
