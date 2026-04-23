using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Runtime;

namespace BitNetSharp.Runtime.Tests;

public sealed class InProcessBitNetModelTests
{
    [Fact]
    public void InProcessBitNetModel_loads_v1_blob()
    {
        var cfg = MinimalConfig();
        var options = SmallOptions();
        var seed = new BitNetPaperModel(options, cfg, seed: 1);
        var flat = FlatParameterPack.Pack(seed.Transformer);
        var blob = WeightBlobCodec.Encode(version: 7L, flat);

        using var runtime = InProcessBitNetModel.LoadFromBytes(blob, options, cfg);

        Assert.Equal(7L, runtime.WeightVersion);

        var response = runtime.GenerateResponse("hello", maxOutputTokens: 2);
        Assert.NotNull(response);
        // Deterministic path must produce *some* response object; text may be empty
        // for this tiny vocab, but the diagnostics list must exist.
        Assert.NotNull(response.Diagnostics);
    }

    [Fact]
    public void InProcessBitNetModel_rejects_magic_mismatch()
    {
        var cfg = MinimalConfig();
        var options = SmallOptions();
        var garbage = new byte[WeightBlobCodec.HeaderSize + 16];
        // Leave magic = 0 -> will not match 0x54475742.

        var ex = Assert.Throws<ArgumentException>(() =>
            InProcessBitNetModel.LoadFromBytes(garbage, options, cfg));
        Assert.Contains("Magic", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void InProcessBitNetModel_rejects_weight_count_mismatch()
    {
        var cfg = MinimalConfig();
        var options = SmallOptions();
        // Encode only 2 floats so header is valid but count doesn't match config expectations.
        var blob = WeightBlobCodec.Encode(version: 1, new float[] { 0.0f, 0.0f });

        var ex = Assert.Throws<ArgumentException>(() =>
            InProcessBitNetModel.LoadFromBytes(blob, options, cfg));
        Assert.Contains("expects", ex.Message);
    }

    [Fact]
    public async Task InProcessBitNetModel_loads_v1_blob_from_disk()
    {
        var cfg = MinimalConfig();
        var options = SmallOptions();
        var seed = new BitNetPaperModel(options, cfg, seed: 2);
        var flat = FlatParameterPack.Pack(seed.Transformer);
        var blob = WeightBlobCodec.Encode(version: 42L, flat);

        var path = Path.Combine(Path.GetTempPath(), $"bnrt-{Guid.NewGuid():N}.blob");
        await File.WriteAllBytesAsync(path, blob);
        try
        {
            using var runtime = await InProcessBitNetModel.LoadAsync(path, options, cfg);
            Assert.Equal(42L, runtime.WeightVersion);
            Assert.Equal(seed.ModelId, runtime.ModelId);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    // Tokenizer adds 3 special tokens to the user-supplied vocab (BOS/EOS/PAD),
    // so VocabSize = Vocabulary.Count + 3.
    private static BitNetConfig MinimalConfig() => new(
        vocabSize: 11,
        dimension: 16,
        hiddenDimension: 32,
        layerCount: 1,
        headCount: 2,
        maxSequenceLength: 16);

    private static BitNetOptions SmallOptions() => new(
        Vocabulary: new[] { "hello", "world", "how", "are", "you", "ok", "fine", "model" },
        Verbosity: VerbosityLevel.Normal,
        MaxResponseTokens: 4);
}
