using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp.Core;

/// <summary>
/// Compact binary store for a trained TruckMate SLM checkpoint shipped to
/// the phone. Holds: model architecture (BitNetConfig fields), the
/// vocabulary (so the on-device tokenizer can be rebuilt without a side
/// channel), and the FlatParameterPack float[] in a single file.
///
/// <para>
/// Format <c>tmv1</c> (little-endian):
/// <list type="bullet">
///   <item><c>uint32 magic = 0x31564D54</c> ("TMV1" little-endian)</item>
///   <item><c>uint32 version = 1</c></item>
///   <item><c>int32 vocab_size</c></item>
///   <item><c>int32 dimension</c></item>
///   <item><c>int32 hidden_dimension</c></item>
///   <item><c>int32 layer_count</c></item>
///   <item><c>int32 head_count</c></item>
///   <item><c>int32 max_sequence_length</c></item>
///   <item><c>int32 kv_head_count</c></item>
///   <item><c>int32 vocab_token_count</c></item>
///   <item>For each token: <c>uint16 utf8_len; byte[] utf8</c></item>
///   <item><c>int32 flat_param_count</c></item>
///   <item><c>float[flat_param_count]</c></item>
/// </list>
/// </para>
///
/// <para>
/// The vocab list is the BitNetPaperModel-compatible form (specials
/// <c>&lt;bos&gt;</c>, <c>&lt;eos&gt;</c>, <c>&lt;unk&gt;</c> already at
/// the front). The parameter vector follows
/// <see cref="BitNetSharp.Core.Training.FlatParameterPack"/>'s canonical
/// order so on-device load can pass the vector straight to
/// <c>FlatParameterPack.Unpack</c>.
/// </para>
///
/// <para>
/// Lives in Core (not Distributed.Contracts) so the MAUI on-device
/// loader can read it without pulling in the full coordinator stack.
/// </para>
/// </summary>
public static class TruckMateModelStore
{
    private const uint Magic = 0x31564D54u; // 'T','M','V','1' little-endian
    private const uint FormatVersion = 1u;

    /// <summary>
    /// Header fields parsed from the on-disk store. Sufficient to
    /// reconstruct a <c>BitNetConfig</c> matching the trained model.
    /// </summary>
    public sealed record Header(
        int VocabSize,
        int Dimension,
        int HiddenDimension,
        int LayerCount,
        int HeadCount,
        int MaxSequenceLength,
        int KvHeadCount,
        IReadOnlyList<string> Vocabulary);

    /// <summary>
    /// Writes the checkpoint to <paramref name="path"/>. Overwrites any
    /// existing file. Caller is responsible for ensuring
    /// <paramref name="vocabulary"/>.Count equals
    /// <paramref name="vocabSize"/> and that
    /// <paramref name="flatParameters"/>.Length matches what
    /// <c>FlatParameterPack.ComputeLength</c> would return for the
    /// declared shape.
    /// </summary>
    public static void Save(
        string path,
        int vocabSize,
        int dimension,
        int hiddenDimension,
        int layerCount,
        int headCount,
        int maxSequenceLength,
        int kvHeadCount,
        IReadOnlyList<string> vocabulary,
        float[] flatParameters)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(flatParameters);

        if (vocabulary.Count != vocabSize)
        {
            throw new ArgumentException(
                $"Vocabulary count {vocabulary.Count} does not match declared vocabSize {vocabSize}.",
                nameof(vocabulary));
        }

        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);
        bw.Write(Magic);
        bw.Write(FormatVersion);
        bw.Write(vocabSize);
        bw.Write(dimension);
        bw.Write(hiddenDimension);
        bw.Write(layerCount);
        bw.Write(headCount);
        bw.Write(maxSequenceLength);
        bw.Write(kvHeadCount);
        bw.Write(vocabulary.Count);
        foreach (var token in vocabulary)
        {
            var bytes = Encoding.UTF8.GetBytes(token ?? string.Empty);
            if (bytes.Length > ushort.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Vocabulary entry '{token}' exceeds 65535 bytes after UTF-8 encoding.");
            }
            bw.Write((ushort)bytes.Length);
            bw.Write(bytes);
        }
        bw.Write(flatParameters.Length);
        // Bulk-write the float buffer as raw bytes for speed.
        var span = MemoryMarshal.AsBytes(new ReadOnlySpan<float>(flatParameters));
        bw.Write(span);
    }

    /// <summary>
    /// Reads the header (config + vocabulary) and the flat parameter
    /// vector from <paramref name="path"/>. Throws on magic/version
    /// mismatch or truncated payload.
    /// </summary>
    public static (Header Header, float[] FlatParameters) Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

        var magic = br.ReadUInt32();
        if (magic != Magic)
        {
            throw new InvalidDataException($"Not a TruckMate model store (magic 0x{magic:X8}).");
        }
        var version = br.ReadUInt32();
        if (version != FormatVersion)
        {
            throw new InvalidDataException($"Unsupported TruckMate model store version {version}.");
        }

        var vocabSize = br.ReadInt32();
        var dimension = br.ReadInt32();
        var hiddenDimension = br.ReadInt32();
        var layerCount = br.ReadInt32();
        var headCount = br.ReadInt32();
        var maxSequenceLength = br.ReadInt32();
        var kvHeadCount = br.ReadInt32();
        var tokenCount = br.ReadInt32();
        if (tokenCount != vocabSize)
        {
            throw new InvalidDataException(
                $"Header token count {tokenCount} disagrees with vocab_size {vocabSize}.");
        }

        var vocab = new string[tokenCount];
        for (var i = 0; i < tokenCount; i++)
        {
            var len = br.ReadUInt16();
            var bytes = br.ReadBytes(len);
            if (bytes.Length != len)
            {
                throw new EndOfStreamException(
                    $"Truncated UTF-8 token at index {i}: expected {len} bytes, got {bytes.Length}.");
            }
            vocab[i] = Encoding.UTF8.GetString(bytes);
        }

        var flatLen = br.ReadInt32();
        if (flatLen < 0)
        {
            throw new InvalidDataException($"Negative flat-parameter length {flatLen}.");
        }
        var flat = new float[flatLen];
        var byteCount = flatLen * sizeof(float);
        var dst = MemoryMarshal.AsBytes(new Span<float>(flat));
        var totalRead = 0;
        while (totalRead < byteCount)
        {
            var n = br.Read(dst.Slice(totalRead));
            if (n <= 0)
            {
                throw new EndOfStreamException(
                    $"Truncated flat-parameter payload: expected {byteCount} bytes, got {totalRead}.");
            }
            totalRead += n;
        }

        var header = new Header(
            vocabSize,
            dimension,
            hiddenDimension,
            layerCount,
            headCount,
            maxSequenceLength,
            kvHeadCount,
            vocab);
        return (header, flat);
    }
}
