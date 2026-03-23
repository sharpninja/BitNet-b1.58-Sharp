using BitNetSharp.Core;
using BitNetSharp.Core.Bucketing;
using System.Reflection;

namespace BitNetSharp.Tests;

public sealed class BitNetPaperCheckpointTests
{
    [Fact]
    public void CheckpointRoundTripPreservesTrainedPromptResponses()
    {
        var model = BitNetPaperModel.CreateForTrainingCorpus(
            BitNetTrainingCorpus.CreateBenchmarkExamples(),
            VerbosityLevel.Quiet);
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        model.Train(examples, epochs: 1);
        var exportFinalNormScale = typeof(BitNetPaperModel).GetMethod("ExportFinalNormScale", BindingFlags.Instance | BindingFlags.NonPublic);
        var exportOutputHeadWeights = typeof(BitNetPaperModel).GetMethod("ExportOutputHeadWeights", BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(exportFinalNormScale);
        Assert.NotNull(exportOutputHeadWeights);

        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-paper-checkpoint-test-{Guid.NewGuid():N}");
        var checkpointPath = Path.Combine(tempDirectory, "model.bitnet.json");
        var bucketSidecarPath = Path.Combine(tempDirectory, "chain-buckets.bin");

        try
        {
            BitNetPaperCheckpoint.Save(model, checkpointPath);
            Assert.False(File.Exists(bucketSidecarPath));
            var reloaded = BitNetPaperCheckpoint.Load(checkpointPath, VerbosityLevel.Quiet);
            var originalFinalNormScale = (float[])exportFinalNormScale.Invoke(model, [])!;
            var reloadedFinalNormScale = (float[])exportFinalNormScale.Invoke(reloaded, [])!;
            var originalOutputHeadWeights = (float[,])exportOutputHeadWeights.Invoke(model, [])!;
            var reloadedOutputHeadWeights = (float[,])exportOutputHeadWeights.Invoke(reloaded, [])!;

            var original = model.GenerateResponse("what does the paper model train on", maxTokens: 16);
            var roundTripped = reloaded.GenerateResponse("what does the paper model train on", maxTokens: 16);

            Assert.Equal(original.ResponseText, roundTripped.ResponseText);
            AssertVectorAlmostEqual(originalFinalNormScale, reloadedFinalNormScale);
            AssertMatrixAlmostEqual(originalOutputHeadWeights, reloadedOutputHeadWeights);
        }
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    [Fact]
    public void CheckpointRoundTripPreservesBucketTableSidecar()
    {
        var model = new BitNetPaperModel(
            new BitNetOptions(
                BitNetTrainingCorpus.CreateDefaultVocabulary(),
                VerbosityLevel.Quiet,
                EnableChainBuckets: true,
                EnableSequenceCompression: true,
                ChainBucketAcceptanceThreshold: 0.91d));
        model.LoadBucketTable(
            new ChainBucketTable(
            [
                new ChainBucket(7, [3, 4, 5], 0.95f),
                new ChainBucket(11, [8, 13], 0.6f)
            ]));

        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-paper-checkpoint-buckets-{Guid.NewGuid():N}");
        var checkpointPath = Path.Combine(tempDirectory, "model.bitnet.json");
        var bucketSidecarPath = Path.Combine(tempDirectory, "chain-buckets.bin");

        try
        {
            BitNetPaperCheckpoint.Save(model, checkpointPath);

            Assert.True(File.Exists(bucketSidecarPath));

            var reloaded = BitNetPaperCheckpoint.Load(checkpointPath, VerbosityLevel.Quiet);

            Assert.True(reloaded.Options.EnableChainBuckets);
            Assert.True(reloaded.Options.EnableSequenceCompression);
            Assert.Equal(0.91d, reloaded.Options.ChainBucketAcceptanceThreshold);

            var bucketTable = Assert.IsType<ChainBucketTable>(reloaded.BucketTable);
            Assert.Equal(2, bucketTable.Count);
            Assert.Collection(
                bucketTable.Buckets,
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
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    private static void AssertMatrixAlmostEqual(float[,] expected, float[,] actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));

        for (var row = 0; row < expected.GetLength(0); row++)
        {
            for (var column = 0; column < expected.GetLength(1); column++)
            {
                Assert.True(
                    MathF.Abs(expected[row, column] - actual[row, column]) <= tolerance,
                    $"Matrix mismatch at [{row}, {column}]: expected {expected[row, column]}, actual {actual[row, column]}.");
            }
        }
    }

    private static void AssertVectorAlmostEqual(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Count, actual.Count);

        for (var index = 0; index < expected.Count; index++)
        {
            Assert.True(
                MathF.Abs(expected[index] - actual[index]) <= tolerance,
                $"Vector mismatch at [{index}]: expected {expected[index]}, actual {actual[index]}.");
        }
    }
}
