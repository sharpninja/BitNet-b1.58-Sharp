using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Tests;

public sealed class BucketRecallHeatMapSerializerTests
{
    [Fact]
    public void RoundTrip_PreservesAllCountersAndTransitions()
    {
        var original = new BucketRecallHeatMap(64);

        original.RecordChainAttempt(0, [5, 10, 15], speculativeStartIndex: 1);
        original.RecordChainAttempt(0, [5, 10, 15], speculativeStartIndex: 1);
        original.RecordTokenAccepted(0, 10);
        original.RecordChainAccepted(0);
        original.RecordChainAccepted(3);
        original.ResetGenerationState();
        original.RecordChainAccepted(0);
        original.RecordChainAccepted(3);

        using var stream = new MemoryStream();
        BucketRecallHeatMapSerializer.Serialize(original, stream);
        stream.Position = 0;
        var restored = BucketRecallHeatMapSerializer.Deserialize(stream);

        Assert.Equal(original.VocabSize, restored.VocabSize);
        Assert.Equal(original.GetAttemptCount(10), restored.GetAttemptCount(10));
        Assert.Equal(original.GetAttemptCount(15), restored.GetAttemptCount(15));
        Assert.Equal(original.GetAcceptCount(10), restored.GetAcceptCount(10));
        Assert.Equal(original.GetChainAttemptCount(0), restored.GetChainAttemptCount(0));
        Assert.Equal(original.GetChainAcceptCount(0), restored.GetChainAcceptCount(0));
        Assert.Equal(original.GetChainAcceptCount(3), restored.GetChainAcceptCount(3));
        Assert.Equal(original.GetTransitionCount(0, 3), restored.GetTransitionCount(0, 3));
        Assert.Equal(2, restored.GetTransitionCount(0, 3));
    }

    [Fact]
    public void Deserialize_RejectsCorruptedChecksum()
    {
        var original = new BucketRecallHeatMap(16);
        original.RecordChainAttempt(0, [1, 2], speculativeStartIndex: 0);

        using var stream = new MemoryStream();
        BucketRecallHeatMapSerializer.Serialize(original, stream);
        var bytes = stream.ToArray();
        bytes[20] ^= 0x01;

        using var corrupted = new MemoryStream(bytes);
        var exception = Assert.Throws<InvalidDataException>(() => BucketRecallHeatMapSerializer.Deserialize(corrupted));
        Assert.Contains("CRC32", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Deserialize_RejectsInvalidMagic()
    {
        var original = new BucketRecallHeatMap(16);

        using var stream = new MemoryStream();
        BucketRecallHeatMapSerializer.Serialize(original, stream);
        var bytes = stream.ToArray();
        bytes[0] = (byte)'X';

        // Recompute CRC32 so the checksum doesn't fail first.
        var crc = ComputeCrc32(bytes.AsSpan(0, bytes.Length - 4));
        BitConverter.TryWriteBytes(bytes.AsSpan(bytes.Length - 4), crc);

        using var corrupted = new MemoryStream(bytes);
        var exception = Assert.Throws<InvalidDataException>(() => BucketRecallHeatMapSerializer.Deserialize(corrupted));
        Assert.Contains("header", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Deserialize_RejectsVersionMismatch()
    {
        var original = new BucketRecallHeatMap(16);

        using var stream = new MemoryStream();
        BucketRecallHeatMapSerializer.Serialize(original, stream);
        var bytes = stream.ToArray();
        // Version is at offset 4 (2 bytes LE). Set to 99.
        bytes[4] = 99;
        bytes[5] = 0;

        var crc = ComputeCrc32(bytes.AsSpan(0, bytes.Length - 4));
        BitConverter.TryWriteBytes(bytes.AsSpan(bytes.Length - 4), crc);

        using var corrupted = new MemoryStream(bytes);
        var exception = Assert.Throws<InvalidDataException>(() => BucketRecallHeatMapSerializer.Deserialize(corrupted));
        Assert.Contains("version", exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RoundTripWithEmptyHeatMap()
    {
        var original = new BucketRecallHeatMap(32);

        using var stream = new MemoryStream();
        BucketRecallHeatMapSerializer.Serialize(original, stream);
        stream.Position = 0;
        var restored = BucketRecallHeatMapSerializer.Deserialize(stream);

        Assert.Equal(32, restored.VocabSize);
        Assert.Equal(0, restored.GetAttemptCount(0));
        Assert.Equal(0, restored.GetChainAcceptCount(0));
        Assert.Equal(0, restored.GetTransitionCount(0, 0));
    }

    [Fact]
    public void FileRoundTrip_PreservesData()
    {
        var original = new BucketRecallHeatMap(32);
        original.RecordChainAttempt(5, [1, 2, 3], 1);
        original.RecordTokenAccepted(5, 2);
        original.RecordChainAccepted(5);

        var tempPath = Path.Combine(Path.GetTempPath(), $"heatmap-test-{Guid.NewGuid():N}.bin");
        try
        {
            BucketRecallHeatMapSerializer.Save(original, tempPath);
            var restored = BucketRecallHeatMapSerializer.Load(tempPath);

            Assert.Equal(32, restored.VocabSize);
            Assert.Equal(1, restored.GetAttemptCount(2));
            Assert.Equal(1, restored.GetAcceptCount(2));
            Assert.Equal(1, restored.GetChainAcceptCount(5));
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
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
