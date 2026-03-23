using System.Buffers.Binary;
using System.Text;

namespace BitNetSharp.Core.Bucketing;

/// <summary>
/// Provides a compact binary serializer for <see cref="ChainBucketTable"/> sidecar persistence.
/// The wire format is little-endian and follows the repository's <c>chain-buckets.bin</c> v1 layout.
/// </summary>
public static class ChainBucketTableBinarySerializer
{
    private static readonly byte[] MagicHeader = Encoding.ASCII.GetBytes("CHNB");
    private const ushort FormatVersion = 1;
    private const ushort EntryCount = ChainBucketTable.MaxBuckets;
    private const ushort MaxChainLength = BucketMiner.MaxNGramLength;
    private const int HeaderLength = 12;
    private const int FooterLength = 4;

    /// <summary>Saves a chain bucket table to a binary file.</summary>
    public static void Save(ChainBucketTable table, string path)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using var stream = File.Create(path);
        Serialize(table, stream);
    }

    /// <summary>Loads a chain bucket table from a binary file.</summary>
    public static ChainBucketTable Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using var stream = File.OpenRead(path);
        return Deserialize(stream);
    }

    /// <summary>Writes a chain bucket table to a stream using the binary sidecar format.</summary>
    public static void Serialize(ChainBucketTable table, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(destination);

        var bucketsById = new ChainBucket?[EntryCount];
        foreach (var bucket in table.Buckets)
        {
            ValidateBucket(bucket);
            if (bucketsById[bucket.ChainId] is not null)
            {
                throw new InvalidDataException($"Duplicate chain bucket id {bucket.ChainId} is not supported by the binary sidecar format.");
            }

            bucketsById[bucket.ChainId] = bucket;
        }

        using var payload = new MemoryStream();
        using (var writer = new BinaryWriter(payload, Encoding.ASCII, leaveOpen: true))
        {
            writer.Write(MagicHeader);
            writer.Write(FormatVersion);
            writer.Write(EntryCount);
            writer.Write(MaxChainLength);
            writer.Write((ushort)0);

            for (var chainId = 0; chainId < EntryCount; chainId++)
            {
                writer.Write((byte)chainId);
                writer.Write((byte)0);

                var bucket = bucketsById[chainId];
                if (bucket is null)
                {
                    writer.Write((ushort)0);
                    writer.Write(0f);
                    continue;
                }

                writer.Write((ushort)bucket.TokenIds.Length);
                foreach (var tokenId in bucket.TokenIds)
                {
                    writer.Write(tokenId);
                }

                writer.Write(bucket.Confidence);
            }
        }

        var buffer = payload.ToArray();
        var checksum = ComputeCrc32(buffer);
        destination.Write(buffer, 0, buffer.Length);

        Span<byte> checksumBuffer = stackalloc byte[FooterLength];
        BinaryPrimitives.WriteUInt32LittleEndian(checksumBuffer, checksum);
        destination.Write(checksumBuffer);
    }

    /// <summary>Reads a chain bucket table from a stream using the binary sidecar format.</summary>
    public static ChainBucketTable Deserialize(Stream source)
    {
        ArgumentNullException.ThrowIfNull(source);

        using var payload = new MemoryStream();
        source.CopyTo(payload);
        var bytes = payload.ToArray();
        var minimumLength = HeaderLength + (EntryCount * (sizeof(byte) + sizeof(byte) + sizeof(ushort) + sizeof(float))) + FooterLength;
        if (bytes.Length < minimumLength)
        {
            throw new InvalidDataException("Chain bucket table binary payload is too short.");
        }

        var expectedChecksum = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(bytes.Length - FooterLength, FooterLength));
        var actualChecksum = ComputeCrc32(bytes.AsSpan(0, bytes.Length - FooterLength));
        if (actualChecksum != expectedChecksum)
        {
            throw new InvalidDataException("Chain bucket table CRC32 checksum mismatch.");
        }

        using var reader = new BinaryReader(new MemoryStream(bytes, writable: false), Encoding.ASCII, leaveOpen: false);
        var magic = reader.ReadBytes(MagicHeader.Length);
        if (!magic.AsSpan().SequenceEqual(MagicHeader))
        {
            throw new InvalidDataException("Unsupported chain bucket table binary header.");
        }

        var version = reader.ReadUInt16();
        var entryCount = reader.ReadUInt16();
        var maxChainLength = reader.ReadUInt16();
        _ = reader.ReadUInt16();

        if (version != FormatVersion)
        {
            throw new InvalidDataException($"Unsupported chain bucket table version {version}.");
        }

        if (entryCount != EntryCount)
        {
            throw new InvalidDataException($"Unsupported chain bucket entry count {entryCount}. Expected {EntryCount}.");
        }

        if (maxChainLength != MaxChainLength)
        {
            throw new InvalidDataException($"Unsupported max chain length {maxChainLength}. Expected {MaxChainLength}.");
        }

        var buckets = new List<ChainBucket>(entryCount);
        for (var index = 0; index < entryCount; index++)
        {
            var chainId = reader.ReadByte();
            _ = reader.ReadByte();
            var tokenCount = reader.ReadUInt16();

            if (chainId != index)
            {
                throw new InvalidDataException($"Chain bucket entry {index} was encoded with out-of-order chain id {chainId}.");
            }

            if (tokenCount > maxChainLength)
            {
                throw new InvalidDataException(
                    $"Chain bucket {chainId} has invalid token count {tokenCount}. Maximum supported length is {maxChainLength}.");
            }

            var tokenIds = new int[tokenCount];
            for (var tokenIndex = 0; tokenIndex < tokenCount; tokenIndex++)
            {
                tokenIds[tokenIndex] = reader.ReadInt32();
            }

            var confidence = reader.ReadSingle();
            if (tokenCount == 0)
            {
                continue;
            }

            if (tokenCount < BucketMiner.MinNGramLength)
            {
                throw new InvalidDataException(
                    $"Chain bucket {chainId} has invalid token count {tokenCount}. Expected {BucketMiner.MinNGramLength}-{BucketMiner.MaxNGramLength}.");
            }

            buckets.Add(new ChainBucket(chainId, tokenIds, confidence));
        }

        return new ChainBucketTable(buckets);
    }

    private static void ValidateBucket(ChainBucket bucket)
    {
        ArgumentNullException.ThrowIfNull(bucket);

        if (bucket.TokenIds.Length < BucketMiner.MinNGramLength || bucket.TokenIds.Length > BucketMiner.MaxNGramLength)
        {
            throw new InvalidDataException(
                $"Chain bucket {bucket.ChainId} has invalid token count {bucket.TokenIds.Length}. Expected {BucketMiner.MinNGramLength}-{BucketMiner.MaxNGramLength}.");
        }
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
