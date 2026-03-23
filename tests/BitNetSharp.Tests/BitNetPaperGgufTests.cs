using System.Reflection;
using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Bucketing;

namespace BitNetSharp.Tests;

public sealed class BitNetPaperGgufTests
{
    [Fact]
    public void GgufRoundTripPreservesTrainedPromptResponsesAndBucketSidecar()
    {
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        var model = new BitNetPaperModel(
            new BitNetOptions(
                BitNetTrainingCorpus.CreateVocabulary(examples),
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
        model.Train(examples, epochs: 1);

        var exportFinalNormScale = typeof(BitNetPaperModel).GetMethod("ExportFinalNormScale", BindingFlags.Instance | BindingFlags.NonPublic);
        var exportOutputHeadWeights = typeof(BitNetPaperModel).GetMethod("ExportOutputHeadWeights", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(exportFinalNormScale);
        Assert.NotNull(exportOutputHeadWeights);

        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-paper-gguf-test-{Guid.NewGuid():N}");
        var ggufPath = Path.Combine(tempDirectory, "model.gguf");
        var bucketSidecarPath = Path.Combine(tempDirectory, "model.chain-buckets.bin");

        try
        {
            BitNetPaperGguf.Save(model, ggufPath);
            Assert.True(File.Exists(bucketSidecarPath));

            var reloaded = BitNetPaperGguf.Load(ggufPath, VerbosityLevel.Quiet);
            var originalFinalNormScale = (float[])exportFinalNormScale.Invoke(model, [])!;
            var reloadedFinalNormScale = (float[])exportFinalNormScale.Invoke(reloaded, [])!;
            var originalOutputHeadWeights = (float[,])exportOutputHeadWeights.Invoke(model, [])!;
            var reloadedOutputHeadWeights = (float[,])exportOutputHeadWeights.Invoke(reloaded, [])!;

            var original = model.GenerateResponse("what does the paper model train on", maxTokens: 16);
            var roundTripped = reloaded.GenerateResponse("what does the paper model train on", maxTokens: 16);

            Assert.Equal(original.ResponseText, roundTripped.ResponseText);
            AssertVectorAlmostEqual(originalFinalNormScale, reloadedFinalNormScale);
            AssertMatrixAlmostEqual(originalOutputHeadWeights, reloadedOutputHeadWeights);
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

    [Fact]
    public void HostedAgentModelFactoryLoadsRepoAuthoredGguf()
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-paper-gguf-factory-{Guid.NewGuid():N}");
        var ggufPath = Path.Combine(tempDirectory, "model.gguf");

        try
        {
            BitNetPaperGguf.Save(BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet), ggufPath);

            using var hostedModel = HostedAgentModelFactory.Create(ggufPath, VerbosityLevel.Quiet);
            var bitNetModel = Assert.IsType<BitNetHostedAgentModel>(hostedModel);

            Assert.Equal(HostedAgentModelFactory.DefaultModelId, bitNetModel.ModelId);
            Assert.False(string.IsNullOrWhiteSpace(bitNetModel.Model.GenerateResponse("how are you hosted", maxTokens: 4).ResponseText));
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
    public void GgufLoadRejectsInvalidHeader()
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-paper-gguf-invalid-{Guid.NewGuid():N}");
        var ggufPath = Path.Combine(tempDirectory, "invalid.gguf");

        try
        {
            Directory.CreateDirectory(tempDirectory);
            File.WriteAllBytes(ggufPath, [1, 2, 3, 4, 5]);

            Assert.Throws<InvalidDataException>(() => BitNetPaperGguf.Load(ggufPath, VerbosityLevel.Quiet));
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
