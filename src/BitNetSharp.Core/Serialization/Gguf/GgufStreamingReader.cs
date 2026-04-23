using System.Runtime.InteropServices;
using System.Text;

namespace BitNetSharp.Core.Serialization.Gguf;

/// <summary>
/// Describes one tensor in a GGUF file without loading its payload.
/// <paramref name="TensorType"/> identifies the on-disk encoding (see
/// <see cref="GgufTensorTypes"/>). <paramref name="PayloadBytes"/> is the
/// exact byte length of the tensor's payload so callers can sanity-check
/// their reads. The absolute offset is recorded so callers can lazily stream
/// exactly the tensors they need in the order that suits them, instead of
/// materializing the full tensor set up-front (which would OOM for
/// multi-billion-parameter BitNet models).
/// </summary>
internal sealed record GgufTensorInfo(
    string Name,
    IReadOnlyList<int> Dimensions,
    uint TensorType,
    long AbsoluteOffset,
    long PayloadBytes,
    long ElementCount);

/// <summary>
/// Streaming counterpart to <see cref="GgufReader"/>. Reads the header,
/// metadata block, and tensor info table up-front (small, bounded), then
/// exposes per-tensor <see cref="ReadMatrix"/> / <see cref="ReadVector"/> /
/// <see cref="ReadPackedTernary"/> APIs that allocate only one tensor's
/// buffer at a time and discard it when the caller releases the reference.
/// </summary>
internal sealed class GgufStreamingReader : IDisposable
{
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("GGUF");
    private const uint SupportedVersion = 3;
    private const uint DefaultAlignment = 32;

    private readonly FileStream _stream;
    private readonly BinaryReader _reader;
    private bool _disposed;

    private GgufStreamingReader(FileStream stream, IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufTensorInfo> tensors)
    {
        _stream = stream;
        _reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        Metadata = metadata;
        Tensors = tensors;
    }

    public IReadOnlyDictionary<string, object> Metadata { get; }
    public IReadOnlyList<GgufTensorInfo> Tensors { get; }

    public static GgufStreamingReader Open(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var stream = File.OpenRead(path);
        try
        {
            using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
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
            for (var i = 0; i < metadataCount; i++)
            {
                var key = ReadString(reader);
                metadata[key] = ReadMetadataValue(reader);
            }

            var relativeInfos = new List<(string Name, int[] Dimensions, uint TensorType, ulong RelativeOffset)>(tensorCount);
            for (var i = 0; i < tensorCount; i++)
            {
                var name = ReadString(reader);
                var rank = checked((int)reader.ReadUInt32());
                var dims = new int[rank];
                for (var d = 0; d < rank; d++)
                {
                    dims[d] = ReadCount(reader.ReadUInt64(), "tensor dimension");
                }

                var tensorType = reader.ReadUInt32();
                relativeInfos.Add((name, dims, tensorType, reader.ReadUInt64()));
            }

            AlignStream(stream, GetAlignment(metadata));
            var dataStart = stream.Position;

            var tensors = new List<GgufTensorInfo>(relativeInfos.Count);
            for (var i = 0; i < relativeInfos.Count; i++)
            {
                var (name, dims, tensorType, relOffset) = relativeInfos[i];

                var elementCount = 1L;
                foreach (var d in dims)
                {
                    elementCount = checked(elementCount * d);
                }

                var payloadBytes = ComputePayloadBytes(tensorType, dims, elementCount, name);
                var absOffset = checked(dataStart + (long)relOffset);
                tensors.Add(new GgufTensorInfo(name, dims, tensorType, absOffset, payloadBytes, elementCount));
            }

            return new GgufStreamingReader(stream, metadata, tensors);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Reads a rank-2 FP32 tensor directly from disk into a freshly allocated
    /// <c>float[rows, columns]</c>. Peak resident memory is
    /// O(rows * columns * 4) while the caller holds the reference; releasing
    /// the reference between calls lets the GC reclaim the buffer before the
    /// next tensor is read.
    /// </summary>
    public float[,] ReadMatrix(GgufTensorInfo info, int expectedRows, int expectedColumns)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(info);

        if (info.TensorType != GgufTensorTypes.Float32)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' has type {info.TensorType}; ReadMatrix only supports Float32 (type {GgufTensorTypes.Float32}).");
        }

        if (info.Dimensions.Count != 2)
        {
            throw new InvalidDataException($"GGUF tensor '{info.Name}' must be rank 2.");
        }

        if (info.Dimensions[0] != expectedRows || info.Dimensions[1] != expectedColumns)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' expected shape [{expectedRows}, {expectedColumns}] but found [{info.Dimensions[0]}, {info.Dimensions[1]}].");
        }

        _stream.Position = info.AbsoluteOffset;
        var matrix = new float[expectedRows, expectedColumns];
        var rowBuffer = new float[expectedColumns];
        var rowBytes = MemoryMarshal.AsBytes(rowBuffer.AsSpan());

        for (var row = 0; row < expectedRows; row++)
        {
            ReadFullBuffer(rowBytes, info.Name);
            for (var column = 0; column < expectedColumns; column++)
            {
                matrix[row, column] = rowBuffer[column];
            }
        }

        return matrix;
    }

    /// <summary>
    /// Reads a rank-1 FP32 tensor into a freshly allocated float[].
    /// </summary>
    public float[] ReadVector(GgufTensorInfo info, int expectedLength)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(info);

        if (info.TensorType != GgufTensorTypes.Float32)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' has type {info.TensorType}; ReadVector only supports Float32 (type {GgufTensorTypes.Float32}).");
        }

        if (info.Dimensions.Count != 1)
        {
            throw new InvalidDataException($"GGUF tensor '{info.Name}' must be rank 1.");
        }

        if (info.Dimensions[0] != expectedLength)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' expected length {expectedLength} but found {info.Dimensions[0]}.");
        }

        _stream.Position = info.AbsoluteOffset;
        var values = new float[expectedLength];
        ReadFullBuffer(MemoryMarshal.AsBytes(values.AsSpan()), info.Name);
        return values;
    }

    /// <summary>
    /// Reads a v2 BitNetSharp packed-ternary rank-2 tensor. Payload layout on
    /// disk: <c>[float32 gamma][byte[] packed]</c> where packed has
    /// <c>outputDim * ((inputDim + 4) / 5)</c> bytes (5 trits per byte,
    /// base-3). Returns the raw packed byte array plus the per-tensor gamma.
    /// The packed bytes are in the same layout <see cref="Layers.BitLinear"/>
    /// stores internally, so the caller can hand them directly to
    /// <see cref="Layers.BitLinear.ImportPacked"/> without unpacking.
    /// </summary>
    public (byte[] Packed, float Gamma) ReadPackedTernary(GgufTensorInfo info, int expectedOutputDim, int expectedInputDim)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(info);

        if (info.TensorType != GgufTensorTypes.BitNetSharpPackedTernary)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' has type {info.TensorType}; ReadPackedTernary only supports BitNetSharpPackedTernary (type {GgufTensorTypes.BitNetSharpPackedTernary}).");
        }

        if (info.Dimensions.Count != 2)
        {
            throw new InvalidDataException($"GGUF tensor '{info.Name}' must be rank 2.");
        }

        if (info.Dimensions[0] != expectedOutputDim || info.Dimensions[1] != expectedInputDim)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' expected shape [{expectedOutputDim}, {expectedInputDim}] but found [{info.Dimensions[0]}, {info.Dimensions[1]}].");
        }

        var packedStride = (expectedInputDim + 4) / 5;
        var packedLength = checked(packedStride * expectedOutputDim);
        var expectedPayload = checked(sizeof(float) + (long)packedLength);
        if (info.PayloadBytes != expectedPayload)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{info.Name}' payload is {info.PayloadBytes} bytes, expected {expectedPayload}.");
        }

        _stream.Position = info.AbsoluteOffset;
        Span<byte> gammaBuffer = stackalloc byte[sizeof(float)];
        ReadFullBuffer(gammaBuffer, info.Name);
        var gamma = BitConverter.ToSingle(gammaBuffer);

        var packed = new byte[packedLength];
        ReadFullBuffer(packed.AsSpan(), info.Name);
        return (packed, gamma);
    }

    private void ReadFullBuffer(Span<byte> buffer, string tensorName)
    {
        var total = 0;
        while (total < buffer.Length)
        {
            var read = _stream.Read(buffer[total..]);
            if (read == 0)
            {
                throw new InvalidDataException($"Unexpected end of GGUF payload while reading tensor '{tensorName}'.");
            }

            total += read;
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _reader.Dispose();
        _stream.Dispose();
    }

    private static long ComputePayloadBytes(uint tensorType, int[] dims, long elementCount, string tensorName)
    {
        switch (tensorType)
        {
            case GgufTensorTypes.Float32:
                return checked(elementCount * sizeof(float));
            case GgufTensorTypes.BitNetSharpPackedTernary:
                if (dims.Length != 2)
                {
                    throw new InvalidDataException(
                        $"GGUF tensor '{tensorName}' is BitNetSharpPackedTernary but has rank {dims.Length}; must be rank 2.");
                }

                var outputDim = dims[0];
                var inputDim = dims[1];
                var stride = (inputDim + 4) / 5;
                return checked(sizeof(float) + (long)stride * outputDim);
            default:
                throw new InvalidDataException(
                    $"GGUF tensor '{tensorName}' has unsupported ggml_type {tensorType}. Only Float32 ({GgufTensorTypes.Float32}) and BitNetSharpPackedTernary ({GgufTensorTypes.BitNetSharpPackedTernary}) are supported.");
        }
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
