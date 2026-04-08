using System.Buffers.Binary;
using System.Text;

namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// Provides a compact binary serializer for <see cref="BucketRecallHeatMap"/> sidecar persistence.
/// The wire format is little-endian and follows the repository's <c>recall-heatmap.bin</c> v1 layout.
/// </summary>
public static class BucketRecallHeatMapSerializer
{
    private static readonly byte[] MagicHeader = Encoding.ASCII.GetBytes("BRHM");
    private const ushort FormatVersion = 1;
    private const int HeaderLength = 12;
    private const int FooterLength = 4;

    /// <summary>Saves a recall heat map to a binary file.</summary>
    public static void Save(BucketRecallHeatMap heatMap, string path)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var stream = File.Create(path);
        Serialize(heatMap, stream);
    }

    /// <summary>Loads a recall heat map from a binary file.</summary>
    public static BucketRecallHeatMap Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var stream = File.OpenRead(path);
        return Deserialize(stream);
    }

    /// <summary>Writes a recall heat map to a stream using the binary sidecar format.</summary>
    public static void Serialize(BucketRecallHeatMap heatMap, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(heatMap);
        ArgumentNullException.ThrowIfNull(destination);

        var counters = heatMap.ExportCounters();

        using var payload = new MemoryStream();
        using (var writer = new BinaryWriter(payload, Encoding.ASCII, leaveOpen: true))
        {
            writer.Write(MagicHeader);
            writer.Write(FormatVersion);
            writer.Write((ushort)0);
            writer.Write(counters.VocabSize);

            for (var i = 0; i < counters.VocabSize; i++)
            {
                writer.Write(counters.AttemptCounts[i]);
            }

            for (var i = 0; i < counters.VocabSize; i++)
            {
                writer.Write(counters.AcceptCounts[i]);
            }

            for (var i = 0; i < ChainBucketTable.MaxBuckets; i++)
            {
                writer.Write(counters.ChainAttemptCounts[i]);
            }

            for (var i = 0; i < ChainBucketTable.MaxBuckets; i++)
            {
                writer.Write(counters.ChainAcceptCounts[i]);
            }

            for (var row = 0; row < ChainBucketTable.MaxBuckets; row++)
            {
                for (var col = 0; col < ChainBucketTable.MaxBuckets; col++)
                {
                    writer.Write(counters.ChainTransitions[row, col]);
                }
            }
        }

        var buffer = payload.ToArray();
        var checksum = ComputeCrc32(buffer);
        destination.Write(buffer, 0, buffer.Length);

        Span<byte> checksumBuffer = stackalloc byte[FooterLength];
        BinaryPrimitives.WriteUInt32LittleEndian(checksumBuffer, checksum);
        destination.Write(checksumBuffer);
    }

    /// <summary>Reads a recall heat map from a stream using the binary sidecar format.</summary>
    public static BucketRecallHeatMap Deserialize(Stream source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using var payload = new MemoryStream();
        source.CopyTo(payload);
        var bytes = payload.ToArray();

        if (bytes.Length < HeaderLength + FooterLength)
        {
            throw new InvalidDataException("Recall heat map binary payload is too short.");
        }

        var expectedChecksum = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(bytes.Length - FooterLength, FooterLength));
        var actualChecksum = ComputeCrc32(bytes.AsSpan(0, bytes.Length - FooterLength));
        if (actualChecksum != expectedChecksum)
        {
            throw new InvalidDataException("Recall heat map CRC32 checksum mismatch.");
        }

        using var reader = new BinaryReader(new MemoryStream(bytes, writable: false), Encoding.ASCII, leaveOpen: false);
        var magic = reader.ReadBytes(MagicHeader.Length);
        if (!magic.AsSpan().SequenceEqual(MagicHeader))
        {
            throw new InvalidDataException("Unsupported recall heat map binary header.");
        }

        var version = reader.ReadUInt16();
        _ = reader.ReadUInt16();
        var vocabSize = reader.ReadInt32();

        if (version != FormatVersion)
        {
            throw new InvalidDataException($"Unsupported recall heat map version {version}.");
        }

        if (vocabSize <= 0)
        {
            throw new InvalidDataException($"Invalid vocab size {vocabSize} in recall heat map.");
        }

        var attemptCounts = new long[vocabSize];
        for (var i = 0; i < vocabSize; i++)
        {
            attemptCounts[i] = reader.ReadInt64();
        }

        var acceptCounts = new long[vocabSize];
        for (var i = 0; i < vocabSize; i++)
        {
            acceptCounts[i] = reader.ReadInt64();
        }

        var chainAttemptCounts = new long[ChainBucketTable.MaxBuckets];
        for (var i = 0; i < ChainBucketTable.MaxBuckets; i++)
        {
            chainAttemptCounts[i] = reader.ReadInt64();
        }

        var chainAcceptCounts = new long[ChainBucketTable.MaxBuckets];
        for (var i = 0; i < ChainBucketTable.MaxBuckets; i++)
        {
            chainAcceptCounts[i] = reader.ReadInt64();
        }

        var chainTransitions = new long[ChainBucketTable.MaxBuckets, ChainBucketTable.MaxBuckets];
        for (var row = 0; row < ChainBucketTable.MaxBuckets; row++)
        {
            for (var col = 0; col < ChainBucketTable.MaxBuckets; col++)
            {
                chainTransitions[row, col] = reader.ReadInt64();
            }
        }

        var counters = new HeatMapCounters(vocabSize, attemptCounts, acceptCounts, chainAttemptCounts, chainAcceptCounts, chainTransitions);
        return BucketRecallHeatMap.FromCounters(counters);
    }

    private static uint ComputeCrc32(ReadOnlySpan<byte> data)
    {
        const uint polynomial = 0xEDB88320u;
        var crc = 0xFFFFFFFFu;

        foreach (var value in data)
        {
            crc ^= value;
            for (var bit = 0; bit < 8; bit++)
            {
                var mask = (crc & 1u) == 0u ? 0u : polynomial;
                crc = (crc >> 1) ^ mask;
            }
        }

        return ~crc;
    }
}
