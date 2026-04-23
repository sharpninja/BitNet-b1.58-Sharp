using System.IO;
using System.Text;
using System.Text.Json;
using BitNetSharp.App.Serve.Framing;

namespace BitNetSharp.Tests.Serve;

public sealed class OllamaStreamWriterTests
{
    [Fact]
    public async Task WriteAsync_EmitsOneLinePerPayloadTerminatedByLineFeed()
    {
        using var ms = new MemoryStream();
        var writer = new OllamaStreamWriter(ms);
        await writer.WriteAsync(new { a = 1 });
        await writer.WriteAsync(new { a = 2 });
        string body = Encoding.UTF8.GetString(ms.ToArray());
        Assert.Equal("{\"a\":1}\n{\"a\":2}\n", body);
    }

    [Fact]
    public async Task WriteAsync_UsesSnakeCasePolicyByDefault()
    {
        using var ms = new MemoryStream();
        var writer = new OllamaStreamWriter(ms);
        await writer.WriteAsync(new { DoneReason = "stop" });
        string body = Encoding.UTF8.GetString(ms.ToArray());
        var doc = JsonDocument.Parse(body.TrimEnd('\n'));
        Assert.Equal("stop", doc.RootElement.GetProperty("done_reason").GetString());
    }
}
