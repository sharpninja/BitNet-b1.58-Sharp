using BitNetSharp.Core;

namespace BitNetSharp.Tests;

public sealed class TraditionalLocalCheckpointTests
{
    [Fact]
    public void CheckpointRoundTripPreservesTrainedPromptResponses()
    {
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        var model = TraditionalLocalModel.CreateForTrainingCorpus(examples, VerbosityLevel.Quiet);
        model.Train(examples, epochs: TraditionalLocalModel.DefaultTrainingEpochs);

        var checkpointPath = Path.Combine(Path.GetTempPath(), $"traditional-local-checkpoint-test-{Guid.NewGuid():N}.json");

        try
        {
            TraditionalLocalCheckpoint.Save(model, checkpointPath);
            var reloaded = TraditionalLocalCheckpoint.Load(checkpointPath, VerbosityLevel.Quiet);

            var original = model.GenerateResponse("what does the paper model train on", maxTokens: 16);
            var roundTripped = reloaded.GenerateResponse("what does the paper model train on", maxTokens: 16);

            Assert.Equal(original.ResponseText, roundTripped.ResponseText);
            Assert.Equal(original.Tokens, roundTripped.Tokens);
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
