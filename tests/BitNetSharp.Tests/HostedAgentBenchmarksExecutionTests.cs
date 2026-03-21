using BitNetSharp.App;
using BitNetSharp.Core;

namespace BitNetSharp.Tests;

[CollectionDefinition("Benchmark environment", DisableParallelization = true)]
public sealed class BenchmarkEnvironmentCollectionDefinition
{
}

[Collection("Benchmark environment")]
public sealed class HostedAgentBenchmarksExecutionTests
{
    [Fact]
    public async Task ResponseBenchmarkExecutesThePaperAlignedQueryPath()
    {
        await WithBenchmarkOptionsAsync(
            new HostedAgentBenchmarkOptions(
                [HostedAgentModelFactory.DefaultModelId],
                "how are you hosted",
                MaxOutputTokens: 3,
                VerbosityLevel.Normal),
            async () =>
            {
                var benchmark = new HostedAgentResponseBenchmarks
                {
                    ModelSpecifier = HostedAgentModelFactory.DefaultModelId
                };

                var response = await benchmark.GenerateResponseForPrompt();

                Assert.Contains("microsoft", response, StringComparison.OrdinalIgnoreCase);
            });
    }

    [Fact]
    public async Task StreamingBenchmarkProducesAtLeastOneUpdate()
    {
        await WithBenchmarkOptionsAsync(
            new HostedAgentBenchmarkOptions(
                [HostedAgentModelFactory.DefaultModelId],
                "how are you hosted",
                MaxOutputTokens: 3,
                VerbosityLevel.Normal),
            async () =>
            {
                var benchmark = new HostedAgentStreamingBenchmarks
                {
                    ModelSpecifier = HostedAgentModelFactory.DefaultModelId
                };

                var updateCount = await benchmark.StreamResponseForPrompt();

                Assert.True(updateCount > 0);
            });
    }

    [Fact]
    public async Task HostBenchmarkBuildsTheConfiguredPaperAlignedModelHost()
    {
        await WithBenchmarkOptionsAsync(
            new HostedAgentBenchmarkOptions(
                [HostedAgentModelFactory.DefaultModelId],
                "how are you hosted",
                MaxOutputTokens: 3,
                VerbosityLevel.Normal),
            () =>
            {
                var benchmark = new HostedAgentHostBenchmarks
                {
                    ModelSpecifier = HostedAgentModelFactory.DefaultModelId
                };

                var modelId = benchmark.BuildAgentHost();

                Assert.Equal(HostedAgentModelFactory.DefaultModelId, modelId);
                return Task.CompletedTask;
            });
    }

    [Fact]
    public async Task TrainingBenchmarkRunsTheSharedTinyLlamaDatasetForTrainableModels()
    {
        await WithBenchmarkOptionsAsync(
            new HostedAgentBenchmarkOptions(
                [HostedAgentModelFactory.DefaultModelId, HostedAgentModelFactory.TraditionalLocalModelId],
                "how are you hosted",
                MaxOutputTokens: 3,
                VerbosityLevel.Normal),
            () =>
            {
                var benchmark = new HostedAgentTrainingBenchmarks
                {
                    ModelSpecifier = HostedAgentModelFactory.TraditionalLocalModelId
                };

                var trainedExamples = benchmark.TrainSelectedModel();

                Assert.Equal(BitNetTrainingCorpus.CreateBenchmarkExamples().Count, trainedExamples);
                return Task.CompletedTask;
            });
    }

    [Fact]
    public async Task TrainingBenchmarkRunsThePaperAlignedTinyLlamaTrainingPath()
    {
        await WithBenchmarkOptionsAsync(
            new HostedAgentBenchmarkOptions(
                [HostedAgentModelFactory.DefaultModelId],
                "how are you hosted",
                MaxOutputTokens: 3,
                VerbosityLevel.Normal),
            () =>
            {
                var benchmark = new HostedAgentTrainingBenchmarks
                {
                    ModelSpecifier = HostedAgentModelFactory.DefaultModelId
                };

                var trainedExamples = benchmark.TrainSelectedModel();

                Assert.Equal(BitNetTrainingCorpus.CreateBenchmarkExamples().Count, trainedExamples);
                return Task.CompletedTask;
            });
    }

    [Fact]
    public void PerplexityEvaluationProducesFiniteValuesForBuiltInModelsAfterTinyLlamaBenchmarkTraining()
    {
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        var validationSamples = BitNetBenchmarkFixtures.WikiText2ValidationSamples.Take(4).ToArray();
        var bitNetModel = BitNetPaperModel.CreateForTrainingCorpus(examples);
        var traditionalModel = TraditionalLocalModel.CreateForTrainingCorpus(examples);

        bitNetModel.Train(examples, epochs: 3);
        traditionalModel.Train(examples, epochs: TraditionalLocalModel.DefaultTrainingEpochs);

        var bitNetPerplexity = bitNetModel.CalculatePerplexity(validationSamples);
        var traditionalPerplexity = traditionalModel.CalculatePerplexity(validationSamples);

        Assert.True(double.IsFinite(bitNetPerplexity));
        Assert.True(double.IsFinite(traditionalPerplexity));
        Assert.True(bitNetPerplexity > 0d);
        Assert.True(traditionalPerplexity > 0d);
    }

    [Fact]
    public void WikiText2ValidationSamplesLoadRepositoryLocalPretokenizedFixture()
    {
        var samples = BitNetBenchmarkFixtures.WikiText2ValidationSamples;

        Assert.Equal(2461, samples.Count);
        Assert.Equal(" = Homarus gammarus = ", samples[0]);
        Assert.Equal(" = = = Television roles = = = ", samples[^1]);
        Assert.All(samples, static sample => Assert.False(string.IsNullOrWhiteSpace(sample)));
    }

    [Fact]
    public void BenchmarkModelConstructionUsesTheTinyLlamaTrainingVocabulary()
    {
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        using var bitNet = HostedAgentModelFactory.Create(HostedAgentModelFactory.DefaultModelId, VerbosityLevel.Quiet, examples);
        using var traditional = HostedAgentModelFactory.Create(HostedAgentModelFactory.TraditionalLocalModelId, VerbosityLevel.Quiet, examples);

        Assert.Equal("tinyllama", ((BitNetHostedAgentModel)bitNet).Model.Tokenizer.Normalize("tinyllama"));
        Assert.Equal("tinyllama", ((TraditionalLocalHostedAgentModel)traditional).Model.Tokenizer.Normalize("tinyllama"));
    }

    [Fact]
    public void BuiltInModelsPreserveTrainedResponsesAcrossCheckpointRoundTrips()
    {
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        var bitNetModel = BitNetPaperModel.CreateForTrainingCorpus(examples, VerbosityLevel.Quiet);
        var traditionalModel = TraditionalLocalModel.CreateForTrainingCorpus(examples, VerbosityLevel.Quiet);

        bitNetModel.Train(examples, epochs: 1);
        traditionalModel.Train(examples, epochs: TraditionalLocalModel.DefaultTrainingEpochs);

        var bitNetRoundTrip = BitNetPaperCheckpoint.ValidateRoundTrip(bitNetModel, "what does the paper model train on");
        var traditionalRoundTrip = TraditionalLocalCheckpoint.ValidateRoundTrip(traditionalModel, "what does the paper model train on");

        Assert.True(bitNetRoundTrip.ResponsesMatch);
        Assert.True(traditionalRoundTrip.ResponsesMatch);
    }

    private static async Task WithBenchmarkOptionsAsync(HostedAgentBenchmarkOptions options, Func<Task> assertion)
    {
        var originalValue = Environment.GetEnvironmentVariable(HostedAgentBenchmarkOptions.EnvironmentVariableName);
        options.StoreInEnvironment();

        try
        {
            await assertion();
        }
        finally
        {
            Environment.SetEnvironmentVariable(HostedAgentBenchmarkOptions.EnvironmentVariableName, originalValue);
        }
    }
}
