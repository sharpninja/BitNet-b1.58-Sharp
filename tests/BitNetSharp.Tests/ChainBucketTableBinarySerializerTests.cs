using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Tests;

public sealed class ChainBucketTableBinarySerializerTests
{
    [Fact]
    public void SerializeRoundTripPreservesBucketEntries()
    {
        var original = new ChainBucketTable(
        [
            new ChainBucket(7, [3, 4, 5], 0.95f),
            new ChainBucket(11, [8, 13], 0.6f)
        ]);

        using var stream = new MemoryStream();
        ChainBucketTableBinarySerializer.Serialize(original, stream);
        stream.Position = 0;

        var roundTripped = ChainBucketTableBinarySerializer.Deserialize(stream);

        Assert.Equal(2, roundTripped.Count);
        Assert.Collection(
            roundTripped.Buckets,
            bucket =>
            {
                Assert.Equal((byte)7, bucket.ChainId);
                Assert.Equal([3, 4, 5], bucket.TokenIds);
                Assert.Equal(0.95f, bucket.Confidence);
            },
            bucket =>
            {
                Assert.Equal((byte)11, bucket.ChainId);
                Assert.Equal([8, 13], bucket.TokenIds);
                Assert.Equal(0.6f, bucket.Confidence);
            });
    }

    [Fact]
    public void DeserializeRejectsCorruptedChecksum()
    {
        var original = new ChainBucketTable([new ChainBucket(7, [3, 4, 5], 0.95f)]);

        using var stream = new MemoryStream();
        ChainBucketTableBinarySerializer.Serialize(original, stream);

        var bytes = stream.ToArray();
        bytes[20] ^= 0x01;

        using var corrupted = new MemoryStream(bytes);
        var exception = Assert.Throws<InvalidDataException>(() => ChainBucketTableBinarySerializer.Deserialize(corrupted));
        Assert.Contains("CRC32", exception.Message, StringComparison.Ordinal);
    }
}
