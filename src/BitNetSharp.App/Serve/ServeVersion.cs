namespace BitNetSharp.App.Serve;

/// <summary>
/// Version string advertised by GET /api/version. Ollama clients use this to
/// decide whether an endpoint speaks the Ollama dialect; the value itself
/// doesn't need to match stock Ollama's numbering.
/// </summary>
internal static class ServeVersion
{
    public const string Current = "bitnetsharp-0.1.0";
}
