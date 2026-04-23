using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using BitNetSharp.Core.Serialization.Gguf;

namespace BitNetSharp.Converter.Tests;

/// <summary>
/// Builds synthetic GGUF byte streams for testing the extended reader.
/// Matches the on-disk layout used by GgufWriter: magic, version=3,
/// counts, metadata entries, tensor infos, alignment, data.
/// </summary>
internal static class SyntheticGguf
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint Version = 3;

    public sealed record Tensor(string Name, int[] Dimensions, GgufTensorType Type, byte[] RawData);

    public static byte[] Build(IReadOnlyDictionary<string, object> metadata, IReadOnlyList<Tensor> tensors)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(Magic);
            writer.Write(Version);
            writer.Write((ulong)tensors.Count);
            writer.Write((ulong)metadata.Count);

            foreach (var (key, value) in metadata)
            {
                WriteString(writer, key);
                WriteMetadataValue(writer, value);
            }

            uint alignment = GetAlignment(metadata);
            ulong[] offsets = ComputeOffsets(tensors, alignment);

            for (int i = 0; i < tensors.Count; i++)
            {
                var t = tensors[i];
                WriteString(writer, t.Name);
                writer.Write((uint)t.Dimensions.Length);
                foreach (var d in t.Dimensions)
                {
                    writer.Write((ulong)d);
                }
                writer.Write((uint)t.Type);
                writer.Write(offsets[i]);
            }

            // Align the stream to alignment
            long aligned = (long)Align((ulong)writer.BaseStream.Position, alignment);
            while (writer.BaseStream.Position < aligned)
            {
                writer.Write((byte)0);
            }

            long dataStart = writer.BaseStream.Position;
            for (int i = 0; i < tensors.Count; i++)
            {
                long target = dataStart + (long)offsets[i];
                while (writer.BaseStream.Position < target)
                {
                    writer.Write((byte)0);
                }
                writer.Write(tensors[i].RawData);
            }
        }

        return stream.ToArray();
    }

    private static ulong[] ComputeOffsets(IReadOnlyList<Tensor> tensors, uint alignment)
    {
        ulong[] offsets = new ulong[tensors.Count];
        ulong current = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            current = Align(current, alignment);
            offsets[i] = current;
            current += (ulong)tensors[i].RawData.Length;
        }
        return offsets;
    }

    private static uint GetAlignment(IReadOnlyDictionary<string, object> metadata)
    {
        if (metadata.TryGetValue("general.alignment", out var value))
        {
            return value switch
            {
                uint u32 => u32,
                int s32 => (uint)s32,
                ulong u64 => (uint)u64,
                long s64 => (uint)s64,
                _ => 32u,
            };
        }
        return 32u;
    }

    private static ulong Align(ulong value, uint alignment)
    {
        if (alignment == 0) return value;
        ulong remainder = value % alignment;
        return remainder == 0 ? value : value + (alignment - remainder);
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
    }

    // Matches GgufMetadataValueType encoding in GgufFile.cs
    private const uint TypeUInt32 = 4;
    private const uint TypeInt32 = 5;
    private const uint TypeFloat32 = 6;
    private const uint TypeBool = 7;
    private const uint TypeString = 8;
    private const uint TypeArray = 9;
    private const uint TypeUInt64 = 10;
    private const uint TypeInt64 = 11;
    private const uint TypeFloat64 = 12;

    private static void WriteMetadataValue(BinaryWriter writer, object value)
    {
        switch (value)
        {
            case uint u32:
                writer.Write(TypeUInt32);
                writer.Write(u32);
                break;
            case int s32:
                writer.Write(TypeInt32);
                writer.Write(s32);
                break;
            case ulong u64:
                writer.Write(TypeUInt64);
                writer.Write(u64);
                break;
            case long s64:
                writer.Write(TypeInt64);
                writer.Write(s64);
                break;
            case float f32:
                writer.Write(TypeFloat32);
                writer.Write(f32);
                break;
            case double f64:
                writer.Write(TypeFloat64);
                writer.Write(f64);
                break;
            case bool b:
                writer.Write(TypeBool);
                writer.Write(b);
                break;
            case string s:
                writer.Write(TypeString);
                WriteString(writer, s);
                break;
            case object[] arr:
                writer.Write(TypeArray);
                uint elemType = InferArrayElementType(arr);
                writer.Write(elemType);
                writer.Write((ulong)arr.Length);
                foreach (var item in arr)
                {
                    WriteArrayElement(writer, elemType, item);
                }
                break;
            default:
                throw new InvalidDataException($"Unsupported test metadata type {value.GetType().FullName}");
        }
    }

    private static uint InferArrayElementType(object[] arr)
    {
        if (arr.Length == 0) return TypeString;
        return arr[0] switch
        {
            string => TypeString,
            uint => TypeUInt32,
            int => TypeInt32,
            float => TypeFloat32,
            _ => throw new InvalidDataException($"Unsupported array element type {arr[0].GetType().FullName}"),
        };
    }

    private static void WriteArrayElement(BinaryWriter writer, uint type, object item)
    {
        switch (type)
        {
            case TypeString:
                WriteString(writer, (string)item);
                break;
            case TypeUInt32:
                writer.Write((uint)item);
                break;
            case TypeInt32:
                writer.Write((int)item);
                break;
            case TypeFloat32:
                writer.Write((float)item);
                break;
            default:
                throw new InvalidDataException($"Unsupported array elem type {type}");
        }
    }
}
