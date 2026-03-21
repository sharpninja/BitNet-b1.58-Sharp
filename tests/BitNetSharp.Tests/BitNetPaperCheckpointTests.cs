using BitNetSharp.Core;

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

        var checkpointPath = Path.Combine(Path.GetTempPath(), $"bitnet-paper-checkpoint-test-{Guid.NewGuid():N}.json");

        try
        {
            BitNetPaperCheckpoint.Save(model, checkpointPath);
            var reloaded = BitNetPaperCheckpoint.Load(checkpointPath, VerbosityLevel.Quiet);

            var original = model.GenerateResponse("what does the paper model train on", maxTokens: 16);
            var roundTripped = reloaded.GenerateResponse("what does the paper model train on", maxTokens: 16);

            Assert.Equal(original.ResponseText, roundTripped.ResponseText);
        }
        finally
        {
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }
    }
}
