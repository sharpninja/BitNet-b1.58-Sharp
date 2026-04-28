using System.IO;
using BitNetSharp.Core;
using Xunit;

namespace BitNetSharp.Tests;

/// <summary>
/// Round-trip tests for the TMV1 binary format. The on-device MAUI
/// loader and the desktop trainer must read/write byte-identical
/// payloads or the phone will load garbage.
/// </summary>
public sealed class TruckMateModelStoreTests : IDisposable
{
    private readonly string _tempDir;

    public TruckMateModelStoreTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "tmstore-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tempDir, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Save_then_Load_round_trips_header_and_flat_vector()
    {
        var path = Path.Combine(_tempDir, "model.tmv1");
        var vocab = new[]
        {
            BitNetTokenizer.BeginToken,
            BitNetTokenizer.EndToken,
            BitNetTokenizer.UnknownToken,
            "hello", "world", "trucker"
        };
        const int VocabSize = 6;
        var flat = new float[1024];
        for (var i = 0; i < flat.Length; i++)
        {
            flat[i] = (float)(Math.Sin(i * 0.1) * 0.5);
        }

        TruckMateModelStore.Save(
            path,
            vocabSize: VocabSize,
            dimension: 64,
            hiddenDimension: 256,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 32,
            kvHeadCount: 4,
            vocabulary: vocab,
            flatParameters: flat);

        Assert.True(File.Exists(path));

        var (header, loadedFlat) = TruckMateModelStore.Load(path);
        Assert.Equal(VocabSize, header.VocabSize);
        Assert.Equal(64, header.Dimension);
        Assert.Equal(256, header.HiddenDimension);
        Assert.Equal(2, header.LayerCount);
        Assert.Equal(4, header.HeadCount);
        Assert.Equal(32, header.MaxSequenceLength);
        Assert.Equal(4, header.KvHeadCount);
        Assert.Equal(vocab.Length, header.Vocabulary.Count);
        for (var i = 0; i < vocab.Length; i++)
        {
            Assert.Equal(vocab[i], header.Vocabulary[i]);
        }
        Assert.Equal(flat.Length, loadedFlat.Length);
        for (var i = 0; i < flat.Length; i++)
        {
            Assert.Equal(flat[i], loadedFlat[i]);
        }
    }

    [Fact]
    public void Load_rejects_wrong_magic()
    {
        var path = Path.Combine(_tempDir, "bogus.tmv1");
        File.WriteAllBytes(path, [0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x00, 0x00, 0x00]);
        Assert.Throws<InvalidDataException>(() => TruckMateModelStore.Load(path));
    }

    [Fact]
    public void Save_throws_on_vocab_count_mismatch()
    {
        var path = Path.Combine(_tempDir, "mismatch.tmv1");
        Assert.Throws<ArgumentException>(() =>
            TruckMateModelStore.Save(
                path,
                vocabSize: 10,
                dimension: 64,
                hiddenDimension: 256,
                layerCount: 2,
                headCount: 4,
                maxSequenceLength: 32,
                kvHeadCount: 4,
                vocabulary: new[] { "a", "b", "c" }, // count=3 but vocabSize=10
                flatParameters: new float[16]));
    }

    [Fact]
    public void Save_handles_unicode_tokens()
    {
        var path = Path.Combine(_tempDir, "unicode.tmv1");
        var vocab = new[] { "<bos>", "<eos>", "<unk>", "café", "naïve", "中文", "🚛" };
        TruckMateModelStore.Save(
            path,
            vocabSize: vocab.Length,
            dimension: 64,
            hiddenDimension: 256,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 32,
            kvHeadCount: 4,
            vocabulary: vocab,
            flatParameters: new float[16]);

        var (header, _) = TruckMateModelStore.Load(path);
        Assert.Equal(vocab.Length, header.Vocabulary.Count);
        Assert.Equal("café", header.Vocabulary[3]);
        Assert.Equal("naïve", header.Vocabulary[4]);
        Assert.Equal("中文", header.Vocabulary[5]);
        Assert.Equal("🚛", header.Vocabulary[6]);
    }
}
