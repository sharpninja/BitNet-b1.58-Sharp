using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BitNetSharp.App.Serve.Framing;

/// <summary>
/// NDJSON framer used by /api/chat and /api/generate. Each payload is
/// serialized and terminated by a single <c>\n</c>. Flushes after every
/// payload so Ollama clients (ollama-python, Open WebUI) see chunks
/// immediately. The writer never emits a half line: the JSON object is fully
/// written, then the newline, then the flush.
/// </summary>
internal sealed class OllamaStreamWriter
{
    private static readonly byte[] Newline = new byte[] { (byte)'\n' };
    private readonly Stream _output;
    private readonly JsonSerializerOptions _options;

    public OllamaStreamWriter(Stream output, JsonSerializerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        _output = output;
        _options = options ?? ServeJson.Options;
    }

    public async Task WriteAsync<T>(T payload, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await JsonSerializer.SerializeAsync(_output, payload, _options, cancellationToken).ConfigureAwait(false);
        await _output.WriteAsync(Newline, cancellationToken).ConfigureAwait(false);
        await _output.FlushAsync(cancellationToken).ConfigureAwait(false);
    }
}
