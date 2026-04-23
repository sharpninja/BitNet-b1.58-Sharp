using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BitNetSharp.App.Serve.Framing;

/// <summary>
/// Server-Sent Events framer used by /v1/chat/completions and /v1/completions.
/// Each payload is emitted as <c>data: &lt;json&gt;\n\n</c>. Completion is
/// marked by a final <c>data: [DONE]\n\n</c> frame (OpenAI SDK requirement).
/// UTF-8 throughout; no BOM.
/// </summary>
internal sealed class SseWriter
{
    private static readonly byte[] DataPrefix = Encoding.UTF8.GetBytes("data: ");
    private static readonly byte[] FrameTerminator = new byte[] { (byte)'\n', (byte)'\n' };
    private static readonly byte[] DoneFrame = Encoding.UTF8.GetBytes("data: [DONE]\n\n");

    private readonly Stream _output;
    private readonly JsonSerializerOptions _options;

    public SseWriter(Stream output, JsonSerializerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        _output = output;
        _options = options ?? ServeJson.Options;
    }

    public async Task WriteAsync<T>(T payload, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await _output.WriteAsync(DataPrefix, cancellationToken).ConfigureAwait(false);
        await JsonSerializer.SerializeAsync(_output, payload, _options, cancellationToken).ConfigureAwait(false);
        await _output.WriteAsync(FrameTerminator, cancellationToken).ConfigureAwait(false);
        await _output.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    public async Task WriteDoneAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await _output.WriteAsync(DoneFrame, cancellationToken).ConfigureAwait(false);
        await _output.FlushAsync(cancellationToken).ConfigureAwait(false);
    }
}
