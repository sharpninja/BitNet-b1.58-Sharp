using System;
using System.Diagnostics;
using System.IO;
using BitNetSharp.Core;
using BitNetSharp.Tests.Logging;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Direct in-process inference test for the real 8B-derived Bonsai BitNet
/// GGUF. Calls <see cref="BitNetPaperModel.GenerateResponse(string, int?)"/>
/// directly (no HTTP, no TestServer, no Ollama wire format) and streams
/// every ILogger line from BitNetPaperModel / BitNetTransformer / BitLinear
/// into the xUnit runner via <see cref="XUnitLogger"/>. Purpose is to surface
/// per-layer forward-pass timings and autoregressive step breakdowns so the
/// BitLinear hot-path bottleneck can be located and regressed against a
/// before/after optimization.
///
/// Auto-skips when <c>data/models/bonsai.bitnetsharp.gguf</c> is absent so
/// CI boxes without the 1.4 GB artifact stay green.
/// </summary>
[Trait(TestCategories.Category, TestCategories.SlowLane)]
public sealed class BonsaiGgufHelloInferenceTests
{
    private const string BonsaiGgufRelativePath = "data/models/bonsai.bitnetsharp.gguf";

    private readonly ITestOutputHelper _output;

    public BonsaiGgufHelloInferenceTests(ITestOutputHelper output)
    {
        _output = output ?? throw new ArgumentNullException(nameof(output));
    }

    [Fact]
    public void GenerateResponse_Hello_OnBonsaiGguf_EmitsPerLayerTimings()
    {
        string? modelPath = ResolveBonsaiGgufPath();
        if (modelPath is null)
        {
            _output.WriteLine($"Skipped: {BonsaiGgufRelativePath} not found from working dir {Directory.GetCurrentDirectory()}.");
            return;
        }

        _output.WriteLine($"Loading Bonsai BitNet GGUF from {modelPath}...");

        using var loggerFactory = XUnitLoggerExtensions.CreateXUnitLoggerFactory(_output, LogLevel.Trace);
        var modelLogger = loggerFactory.CreateLogger<BitNetPaperModel>();

        var loadSw = Stopwatch.StartNew();
        var paperModel = BitNetPaperGguf.Load(modelPath, modelLogger, loggerFactory, VerbosityLevel.Normal);
        loadSw.Stop();
        _output.WriteLine(
            $"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s: layers={paperModel.Config.LayerCount} dim={paperModel.Config.Dimension} vocab={paperModel.Config.VocabSize}");

        _output.WriteLine("Calling GenerateResponse(\"Hello\", maxTokens: 1) directly (no HTTP)...");
        var inferSw = Stopwatch.StartNew();
        var result = paperModel.GenerateResponse("Hello", maxTokens: 1);
        inferSw.Stop();

        Assert.NotNull(result);
        _output.WriteLine(
            $"Inference completed in {inferSw.Elapsed.TotalSeconds:F1}s. tokens={result.Tokens.Count} response_chars={result.ResponseText.Length} response=\"{result.ResponseText}\"");

        Assert.True(result.Tokens.Count >= 1, "Expected at least one generated token for maxTokens=1.");
    }

    private static string? ResolveBonsaiGgufPath()
    {
        string direct = Path.GetFullPath(BonsaiGgufRelativePath);
        if (File.Exists(direct))
        {
            return direct;
        }

        // Tests may run from bin/Debug/net10.0; walk upward to repo root.
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, BonsaiGgufRelativePath);
            if (File.Exists(candidate))
            {
                return candidate;
            }
            dir = dir.Parent;
        }
        return null;
    }
}
