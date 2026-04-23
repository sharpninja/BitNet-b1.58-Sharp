using System.IO;
using System.Text;
using BitNetSharp.App.Serve.Framing;

namespace BitNetSharp.Tests.Serve;

public sealed class SseWriterTests
{
    [Fact]
    public async Task WriteAsync_EmitsDataPrefixAndDoubleLineFeedTerminator()
    {
        using var ms = new MemoryStream();
        var writer = new SseWriter(ms);
        await writer.WriteAsync(new { a = 1 });
        string body = Encoding.UTF8.GetString(ms.ToArray());
        Assert.Equal("data: {\"a\":1}\n\n", body);
    }

    [Fact]
    public async Task WriteDoneAsync_EmitsLiteralSentinelFrame()
    {
        using var ms = new MemoryStream();
        var writer = new SseWriter(ms);
        await writer.WriteAsync(new { a = 1 });
        await writer.WriteDoneAsync();
        string body = Encoding.UTF8.GetString(ms.ToArray());
        Assert.EndsWith("data: [DONE]\n\n", body);
    }
}
