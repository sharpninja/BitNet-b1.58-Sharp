using System.Text;

namespace BitNetSharp.Core.Serialization.Gguf;

internal enum GgufMetadataValueType : uint
{
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12
}

internal sealed record GgufTensor(string Name, IReadOnlyList<int> Dimensions, float[] Data);

internal sealed record GgufDocument(IReadOnlyDictionary<string, object> Metadata, IReadOnlyList<GgufTensor> Tensors);

internal static class GgufWriter
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint Version = 3;
    private const uint Float32TensorType = 0;
    private const uint DefaultAlignment = 32;

    public static void Write(string path, IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufTensor> tensors)
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

    public static void Write(Stream destination, IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufTensor> tensors)
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

            writer.Write(Float32TensorType);
            writer.Write(offsets[index]);
        }

        AlignStream(writer.BaseStream, alignment);
        var dataStart = writer.BaseStream.Position;
        for (var index = 0; index < tensors.Count; index++)
        {
            var targetPosition = dataStart + (long)offsets[index];
            WritePadding(writer.BaseStream, targetPosition - writer.BaseStream.Position);

            foreach (var value in tensors[index].Data)
            {
                writer.Write(value);
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

    private static ulong[] ComputeOffsets(IReadOnlyList<GgufTensor> tensors, uint alignment)
    {
        var offsets = new ulong[tensors.Count];
        ulong currentOffset = 0;
        for (var index = 0; index < tensors.Count; index++)
        {
            currentOffset = Align(currentOffset, alignment);
            offsets[index] = currentOffset;
            currentOffset += checked((ulong)tensors[index].Data.Length * sizeof(float));
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

internal static class GgufReader
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint SupportedVersion = 3;
    private const uint Float32TensorType = 0;
    private const uint DefaultAlignment = 32;

    public static GgufDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var stream = File.OpenRead(path);
        return Read(stream);
    }

    public static GgufDocument Read(Stream source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!source.CanSeek)
        {
            throw new InvalidDataException("GGUF reader requires a seekable stream.");
        }

        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        var magic = reader.ReadBytes(Magic.Length);
        if (magic.Length != Magic.Length || !magic.AsSpan().SequenceEqual(Magic))
        {
            throw new InvalidDataException("Unsupported GGUF header.");
        }

        var version = reader.ReadUInt32();
        if (version != SupportedVersion)
        {
            throw new InvalidDataException($"Unsupported GGUF version {version}.");
        }

        var tensorCount = ReadCount(reader.ReadUInt64(), "tensor");
        var metadataCount = ReadCount(reader.ReadUInt64(), "metadata");
        var metadata = new Dictionary<string, object>(metadataCount, StringComparer.Ordinal);
        for (var index = 0; index < metadataCount; index++)
        {
            var key = ReadString(reader);
            metadata[key] = ReadMetadataValue(reader);
        }

        var tensorInfos = new List<(string Name, int[] Dimensions, ulong Offset)>(tensorCount);
        for (var index = 0; index < tensorCount; index++)
        {
            var name = ReadString(reader);
            var rank = checked((int)reader.ReadUInt32());
            var dimensions = new int[rank];
            for (var dimensionIndex = 0; dimensionIndex < rank; dimensionIndex++)
            {
                dimensions[dimensionIndex] = ReadCount(reader.ReadUInt64(), "tensor dimension");
            }

            var tensorType = reader.ReadUInt32();
            if (tensorType != Float32TensorType)
            {
                throw new InvalidDataException($"Unsupported GGUF tensor type {tensorType}. Only float32 tensors are supported.");
            }

            tensorInfos.Add((name, dimensions, reader.ReadUInt64()));
        }

        AlignStream(source, GetAlignment(metadata));
        var dataStart = source.Position;
        var tensors = tensorInfos
            .Select(info => ReadTensor(reader, dataStart, info.Name, info.Dimensions, info.Offset))
            .ToArray();

        return new GgufDocument(metadata, tensors);
    }

    private static GgufTensor ReadTensor(BinaryReader reader, long dataStart, string name, IReadOnlyList<int> dimensions, ulong offset)
    {
        var elementCount = dimensions.Aggregate(1L, static (product, dimension) => checked(product * dimension));
        var byteCount = checked(elementCount * sizeof(float));
        var absoluteOffset = checked(dataStart + (long)offset);
        var stream = reader.BaseStream;
        if (absoluteOffset < 0 || absoluteOffset + byteCount > stream.Length)
        {
            throw new InvalidDataException($"GGUF tensor '{name}' points outside the tensor data section.");
        }

        stream.Position = absoluteOffset;
        var data = new float[checked((int)elementCount)];
        for (var index = 0; index < data.Length; index++)
        {
            data[index] = reader.ReadSingle();
        }

        return new GgufTensor(name, [.. dimensions], data);
    }

    private static object ReadMetadataValue(BinaryReader reader)
    {
        var valueType = (GgufMetadataValueType)reader.ReadUInt32();
        return valueType switch
        {
            GgufMetadataValueType.UInt32 => reader.ReadUInt32(),
            GgufMetadataValueType.Int32 => reader.ReadInt32(),
            GgufMetadataValueType.UInt64 => reader.ReadUInt64(),
            GgufMetadataValueType.Int64 => reader.ReadInt64(),
            GgufMetadataValueType.Float32 => reader.ReadSingle(),
            GgufMetadataValueType.Float64 => reader.ReadDouble(),
            GgufMetadataValueType.Bool => reader.ReadBoolean(),
            GgufMetadataValueType.String => ReadString(reader),
            _ => throw new InvalidDataException($"Unsupported GGUF metadata type {valueType}.")
        };
    }

    private static string ReadString(BinaryReader reader)
    {
        var length = reader.ReadUInt64();
        if (length > int.MaxValue)
        {
            throw new InvalidDataException($"GGUF string length {length} exceeds supported bounds.");
        }

        var bytes = reader.ReadBytes((int)length);
        if (bytes.Length != (int)length)
        {
            throw new InvalidDataException("Unexpected end of GGUF payload while reading a string.");
        }

        return Encoding.UTF8.GetString(bytes);
    }

    private static int ReadCount(ulong value, string label)
    {
        if (value > int.MaxValue)
        {
            throw new InvalidDataException($"GGUF {label} count {value} exceeds supported bounds.");
        }

        return (int)value;
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
        stream.Position = (long)aligned;
    }
}
