namespace BitNetSharp.App.Serve;

/// <summary>
/// Runtime configuration for the <c>serve</c> subcommand. Parsed from CLI
/// flags in <see cref="ServeCommand"/>. Defaults mirror Ollama's native bind
/// so drop-in clients (Open WebUI, continue.dev, ollama-python) work without
/// extra config.
/// </summary>
public sealed record ServeOptions(
    string Host = "127.0.0.1",
    int Port = 11434,
    bool EnableCors = true,
    long MaxRequestBodyBytes = 16 * 1024 * 1024);
