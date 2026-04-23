using System.Text;

namespace BitNetSharp.Core.Serialization.Gguf;

/// <summary>
/// Tensor type codes recognized by the BitNetSharp streaming GGUF writer/reader.
/// The F32 code matches the upstream ggml type enum. The packed-ternary code is
/// in a reserved BitNetSharp-private range (well above upstream ggml values) so
/// it never collides with future ggml quant types if this file is ever opened
/// by stock llama.cpp (which will reject the type with "unknown ggml_type").
/// </summary>
internal static class GgufTensorTypes
{
    public const uint Float32 = 0;
    // 1001 is out of the GGML reserved range. The v2 format stores
    // [float32 gamma][byte[] packed_trits] per BitLinear tensor.
    public const uint BitNetSharpPackedTernary = 1001;
}

/// <summary>
/// A GGUF tensor described as a (header, deferred-writer) triple.
/// <paramref name="TensorType"/> selects the on-disk encoding (see
/// <see cref="GgufTensorTypes"/>). <paramref name="PayloadBytes"/> is used to
/// compute offsets up-front and to validate the deferred writer; it must equal
/// the exact number of bytes <see cref="WritePayload"/> emits.
/// <see cref="WritePayload"/> is invoked exactly once later, after the stream
/// has been seeked to the tensor's aligned offset.
/// </summary>
internal sealed record GgufStreamingTensor(
    string Name,
    IReadOnlyList<int> Dimensions,
    uint TensorType,
    long PayloadBytes,
    Action<BinaryWriter> WritePayload);

/// <summary>
/// A streaming variant of <see cref="GgufWriter"/> that never requires every
/// tensor's payload to be resident in memory at the same time. The producer
/// only needs to know each tensor's byte length up-front so the offset table
/// can be computed; the actual payload is streamed on demand via each tensor's
/// delegate.
///
/// This lets multi-billion-parameter BitNet models be serialized on machines
/// that could never hold the full weight set in RAM.
/// </summary>
internal static class GgufStreamingWriter
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint Version = 3;
    private const uint DefaultAlignment = 32;

    public static void Write(string path, IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufStreamingTensor> tensors)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var stream = File.Create(path);
        Write(stream, metadata, tensors);
    }

    public static void Write(Stream destination, IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufStreamingTensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(tensors);

        var alignment = GetAlignment(metadata);
        using var writer = new BinaryWriter(destination, Encoding.UTF8, leaveOpen: true);
        var offsets = ComputeOffsets(tensors, alignment);

        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((ulong)tensors.Count);
        writer.Write((ulong)metadata.Count);

        foreach (var (key, value) in metadata)
        {
            WriteString(writer, key);
            WriteMetadataValue(writer, value);
        }

        for (var index = 0; index < tensors.Count; index++)
        {
            var tensor = tensors[index];
            WriteString(writer, tensor.Name);
            writer.Write((uint)tensor.Dimensions.Count);
            foreach (var dimension in tensor.Dimensions)
            {
                writer.Write((ulong)dimension);
            }

            writer.Write(tensor.TensorType);
            writer.Write(offsets[index]);
        }

        AlignStream(writer.BaseStream, alignment);
        var dataStart = writer.BaseStream.Position;
        for (var index = 0; index < tensors.Count; index++)
        {
            var targetPosition = dataStart + (long)offsets[index];
            WritePadding(writer.BaseStream, targetPosition - writer.BaseStream.Position);

            var before = writer.BaseStream.Position;
            tensors[index].WritePayload(writer);
            var written = writer.BaseStream.Position - before;
            var expected = tensors[index].PayloadBytes;
            if (written != expected)
            {
                throw new InvalidOperationException(
                    $"GGUF tensor '{tensors[index].Name}' writer produced {written} bytes, expected {expected}.");
            }
        }
    }

    private static void WriteMetadataValue(BinaryWriter writer, object value)
    {
        switch (value)
        {
            case uint unsignedInt32:
                writer.Write((uint)GgufMetadataValueType.UInt32);
                writer.Write(unsignedInt32);
                break;
            case int signedInt32:
                writer.Write((uint)GgufMetadataValueType.Int32);
                writer.Write(signedInt32);
                break;
            case ulong unsignedInt64:
                writer.Write((uint)GgufMetadataValueType.UInt64);
                writer.Write(unsignedInt64);
                break;
            case long signedInt64:
                writer.Write((uint)GgufMetadataValueType.Int64);
                writer.Write(signedInt64);
                break;
            case float float32:
                writer.Write((uint)GgufMetadataValueType.Float32);
                writer.Write(float32);
                break;
            case double float64:
                writer.Write((uint)GgufMetadataValueType.Float64);
                writer.Write(float64);
                break;
            case bool boolean:
                writer.Write((uint)GgufMetadataValueType.Bool);
                writer.Write(boolean);
                break;
            case string text:
                writer.Write((uint)GgufMetadataValueType.String);
                WriteString(writer, text);
                break;
            default:
                throw new InvalidDataException($"Unsupported GGUF metadata type '{value.GetType().FullName}'.");
        }
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    private static ulong[] ComputeOffsets(IReadOnlyList<GgufStreamingTensor> tensors, uint alignment)
    {
        var offsets = new ulong[tensors.Count];
        ulong currentOffset = 0;
        for (var index = 0; index < tensors.Count; index++)
        {
            currentOffset = Align(currentOffset, alignment);
            offsets[index] = currentOffset;
            currentOffset += checked((ulong)tensors[index].PayloadBytes);
        }

        return offsets;
    }

    private static uint GetAlignment(IReadOnlyDictionary<string, object> metadata)
    {
        if (metadata.TryGetValue("general.alignment", out var value))
        {
            return value switch
            {
                uint unsignedInt32 when unsignedInt32 > 0 => unsignedInt32,
                int signedInt32 when signedInt32 > 0 => (uint)signedInt32,
                ulong unsignedInt64 when unsignedInt64 > 0 && unsignedInt64 <= uint.MaxValue => (uint)unsignedInt64,
                long signedInt64 when signedInt64 > 0 && signedInt64 <= uint.MaxValue => (uint)signedInt64,
                _ => throw new InvalidDataException("GGUF metadata key 'general.alignment' must be a positive integer.")
            };
        }

        return DefaultAlignment;
    }

    private static ulong Align(ulong value, uint alignment)
    {
        if (alignment == 0)
        {
            return value;
        }

        var remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private static void AlignStream(Stream stream, uint alignment)
    {
        var aligned = Align((ulong)stream.Position, alignment);
        WritePadding(stream, (long)aligned - stream.Position);
    }

    private static void WritePadding(Stream stream, long bytes)
    {
        if (bytes <= 0)
        {
            return;
        }

        Span<byte> padding = stackalloc byte[256];
        while (bytes > 0)
        {
            var chunkSize = (int)Math.Min(bytes, padding.Length);
            stream.Write(padding[..chunkSize]);
            bytes -= chunkSize;
        }
    }
}
