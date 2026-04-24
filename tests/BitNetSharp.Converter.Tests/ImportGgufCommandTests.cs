using System;
using System.IO;
using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Converter.Tests;

public sealed class ImportGgufCommandTests : IDisposable
{
    private readonly string _workDir;

    public ImportGgufCommandTests()
    {
        _workDir = Path.Combine(Path.GetTempPath(), "bitnet-import-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_workDir);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_workDir))
            {
                Directory.Delete(_workDir, recursive: true);
            }
        }
        catch (IOException) { /* best effort */ }
    }

    [Fact]
    public void Run_MissingInput_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ImportGgufCommand.Run(new[] { "import-gguf", "--output=foo.gguf" }, VerbosityLevel.Quiet));
        Assert.Contains("--input", ex.Message);
    }

    [Fact]
    public void Run_MissingOutput_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            ImportGgufCommand.Run(new[] { "import-gguf", "--input=foo.gguf" }, VerbosityLevel.Quiet));
        Assert.Contains("--output", ex.Message);
    }

    [Fact]
    public void Run_MissingInputFile_Throws()
    {
        string missing = Path.Combine(_workDir, "not-there.gguf");
        string output = Path.Combine(_workDir, "out.gguf");
        Assert.Throws<FileNotFoundException>(() =>
            ImportGgufCommand.Run(
                new[] { "import-gguf", $"--input={missing}", $"--output={output}" },
                VerbosityLevel.Quiet));
    }

    [Fact]
    public void Run_HappyPath_ProducesReloadableBitNetGguf()
    {
        string input = Path.Combine(_workDir, "mini-qwen3.gguf");
        string output = Path.Combine(_workDir, "out-bitnet.gguf");
        File.WriteAllBytes(input, MiniQwen3GgufFactory.Build());

        int rc = ImportGgufCommand.Run(
            new[] { "import-gguf", $"--input={input}", $"--output={output}" },
            VerbosityLevel.Quiet);

        Assert.Equal(0, rc);
        Assert.True(File.Exists(output), $"Output GGUF {output} not written.");

        var reloaded = BitNetPaperGguf.Load(output, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, VerbosityLevel.Quiet);
        Assert.Equal(MiniQwen3GgufFactory.LayerCount, reloaded.Config.LayerCount);
        Assert.Equal(MiniQwen3GgufFactory.Dim, reloaded.Config.Dimension);
        Assert.Equal(MiniQwen3GgufFactory.Hidden, reloaded.Config.HiddenDimension);
        Assert.Equal(MiniQwen3GgufFactory.HeadCount, reloaded.Config.HeadCount);
        Assert.Equal(MiniQwen3GgufFactory.KvHeadCount, reloaded.Config.KvHeadCount);
    }

    [Fact]
    public void Run_HappyPath_CreatesOutputDirectoryIfMissing()
    {
        string input = Path.Combine(_workDir, "mini-qwen3.gguf");
        string outDir = Path.Combine(_workDir, "nested", "deep");
        string output = Path.Combine(outDir, "bitnet.gguf");
        File.WriteAllBytes(input, MiniQwen3GgufFactory.Build());

        Assert.False(Directory.Exists(outDir));

        int rc = ImportGgufCommand.Run(
            new[] { "import-gguf", $"--input={input}", $"--output={output}" },
            VerbosityLevel.Quiet);

        Assert.Equal(0, rc);
        Assert.True(File.Exists(output));
    }
}
