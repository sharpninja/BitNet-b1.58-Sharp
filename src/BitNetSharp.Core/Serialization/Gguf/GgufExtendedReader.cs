using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace BitNetSharp.Core.Serialization.Gguf;

/// <summary>
/// GGML tensor types recognized by the extended reader.
/// Only F32, F16, and PrismML Q2_0 are supported; other ids surface a named error.
/// </summary>
public enum GgufTensorType : uint
{
    F32 = 0,
    F16 = 1,
    PrismQ2_0 = 42,
}

/// <summary>
/// Raw tensor record: preserves the on-disk dtype and byte payload so callers
/// (e.g. the Prism Q2_0 converter) can choose their own decode path.
/// </summary>
public sealed record GgufRawTensor(
    string Name,
    IReadOnlyList<int> Dimensions,
    GgufTensorType GgmlType,
    byte[] RawData);

public sealed record GgufExtendedDocument(
    IReadOnlyDictionary<string, object> Metadata,
    IReadOnlyList<GgufRawTensor> Tensors);

/// <summary>
/// Reads GGUF v3 files and returns raw tensor bytes plus dtype, supporting F32,
/// F16, and Prism Q2_0 (GGML_TYPE_Q2_0 = 42). Non-Prism Q2_0 / Q4_0 / etc. raise
/// a clear error naming the offending tensor. Array metadata is surfaced as
/// object[]; Bonsai tokenizer metadata arrives via this path.
/// </summary>
public static class GgufExtendedReader
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint SupportedVersion = 3;
    private const uint DefaultAlignment = 32;

    public static GgufExtendedDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = File.OpenRead(path);
        return Read(stream);
    }

    public static GgufExtendedDocument Read(Stream source)
    {
        ArgumentNullException.ThrowIfNull(source);
        if (!source.CanSeek)
        {
            throw new InvalidDataException("GGUF extended reader requires a seekable stream.");
        }

        using var reader = new BinaryReader(source, Encoding.UTF8, leaveOpen: true);
        var magic = reader.ReadBytes(Magic.Length);
        if (magic.Length != Magic.Length || !magic.AsSpan().SequenceEqual(Magic))
        {
            throw new InvalidDataException("Unsupported GGUF header.");
        }

        uint version = reader.ReadUInt32();
        if (version != SupportedVersion)
        {
            throw new InvalidDataException($"Unsupported GGUF version {version}.");
        }

        int tensorCount = ReadCount(reader.ReadUInt64(), "tensor");
        int metadataCount = ReadCount(reader.ReadUInt64(), "metadata");

        var metadata = new Dictionary<string, object>(metadataCount, StringComparer.Ordinal);
        for (int i = 0; i < metadataCount; i++)
        {
            string key = ReadString(reader);
            metadata[key] = ReadMetadataValue(reader);
        }

        var infos = new List<(string Name, int[] Dimensions, GgufTensorType Type, ulong Offset)>(tensorCount);
        for (int i = 0; i < tensorCount; i++)
        {
            string name = ReadString(reader);
            int rank = checked((int)reader.ReadUInt32());
            int[] dims = new int[rank];
            for (int d = 0; d < rank; d++)
            {
                dims[d] = ReadCount(reader.ReadUInt64(), "tensor dimension");
            }
            uint typeId = reader.ReadUInt32();
            if (!IsSupported(typeId))
            {
                throw new InvalidDataException(
                    $"GGUF tensor '{name}' uses unsupported ggml_type {typeId}. Extended reader supports F32, F16, and Prism Q2_0 only.");
            }
            ulong offset = reader.ReadUInt64();
            infos.Add((name, dims, (GgufTensorType)typeId, offset));
        }

        AlignStream(source, GetAlignment(metadata));
        long dataStart = source.Position;

        var tensors = new GgufRawTensor[infos.Count];
        for (int i = 0; i < infos.Count; i++)
        {
            var info = infos[i];
            long elementCount = info.Dimensions.Aggregate(1L, static (p, d) => checked(p * d));
            long byteCount = ByteSizeFor(info.Type, elementCount, info.Name);
            long absolute = checked(dataStart + (long)info.Offset);
            if (absolute < 0 || absolute + byteCount > source.Length)
            {
                throw new InvalidDataException($"GGUF tensor '{info.Name}' points outside the data section.");
            }
            source.Position = absolute;
            byte[] raw = reader.ReadBytes(checked((int)byteCount));
            if (raw.Length != byteCount)
            {
                throw new InvalidDataException($"Unexpected EOF while reading tensor '{info.Name}'.");
            }
            tensors[i] = new GgufRawTensor(info.Name, info.Dimensions, info.Type, raw);
        }

        return new GgufExtendedDocument(metadata, tensors);
    }

    public static void DecodeFloat16(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if (source.Length < destination.Length * 2)
        {
            throw new ArgumentException(
                $"Source requires at least {destination.Length * 2} bytes, got {source.Length}.", nameof(source));
        }
        for (int i = 0; i < destination.Length; i++)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(i * 2, 2));
            destination[i] = (float)BitConverter.UInt16BitsToHalf(bits);
        }
    }

    private static bool IsSupported(uint typeId) =>
        typeId == (uint)GgufTensorType.F32
        || typeId == (uint)GgufTensorType.F16
        || typeId == (uint)GgufTensorType.PrismQ2_0;

    private static long ByteSizeFor(GgufTensorType type, long elementCount, string tensorName)
    {
        return type switch
        {
            GgufTensorType.F32 => checked(elementCount * 4),
            GgufTensorType.F16 => checked(elementCount * 2),
            GgufTensorType.PrismQ2_0 when elementCount % PrismQ2_0.BlockWeights == 0
                => checked(elementCount / PrismQ2_0.BlockWeights * PrismQ2_0.BlockBytes),
            GgufTensorType.PrismQ2_0 => throw new InvalidDataException(
                $"Prism Q2_0 tensor '{tensorName}' element count {elementCount} is not a multiple of {PrismQ2_0.BlockWeights}."),
            _ => throw new InvalidDataException(
                $"GGUF tensor '{tensorName}' uses unsupported ggml_type {(uint)type}."),
        };
    }

    private static object ReadMetadataValue(BinaryReader reader)
    {
        var typeId = reader.ReadUInt32();
        return ReadValueOfType(reader, typeId);
    }

    private static object ReadValueOfType(BinaryReader reader, uint typeId)
    {
        // Matches GgufMetadataValueType internal enum values.
        return typeId switch
        {
            4 => reader.ReadUInt32(),
            5 => reader.ReadInt32(),
            6 => reader.ReadSingle(),
            7 => reader.ReadBoolean(),
            8 => (object)ReadString(reader),
            9 => ReadArray(reader),
            10 => reader.ReadUInt64(),
            11 => reader.ReadInt64(),
            12 => reader.ReadDouble(),
            _ => throw new InvalidDataException($"Unsupported GGUF metadata type {typeId}."),
        };
    }

    private static object[] ReadArray(BinaryReader reader)
    {
        uint elemType = reader.ReadUInt32();
        ulong length = reader.ReadUInt64();
        if (length > int.MaxValue)
        {
            throw new InvalidDataException($"GGUF array length {length} exceeds supported bounds.");
        }
        int len = (int)length;
        var arr = new object[len];
        for (int i = 0; i < len; i++)
        {
            arr[i] = ReadValueOfType(reader, elemType);
        }
        return arr;
    }

    private static string ReadString(BinaryReader reader)
    {
        ulong length = reader.ReadUInt64();
        if (length > int.MaxValue)
        {
            throw new InvalidDataException($"GGUF string length {length} exceeds supported bounds.");
        }
        byte[] bytes = reader.ReadBytes((int)length);
        if (bytes.Length != (int)length)
        {
            throw new InvalidDataException("Unexpected EOF while reading GGUF string.");
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
                uint u32 when u32 > 0 => u32,
                int s32 when s32 > 0 => (uint)s32,
                ulong u64 when u64 > 0 && u64 <= uint.MaxValue => (uint)u64,
                long s64 when s64 > 0 && s64 <= uint.MaxValue => (uint)s64,
                _ => throw new InvalidDataException("GGUF metadata 'general.alignment' must be a positive integer."),
            };
        }
        return DefaultAlignment;
    }

    private static ulong Align(ulong value, uint alignment)
    {
        if (alignment == 0) return value;
        ulong remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private static void AlignStream(Stream stream, uint alignment)
    {
        ulong aligned = Align((ulong)stream.Position, alignment);
        stream.Position = (long)aligned;
    }
}
