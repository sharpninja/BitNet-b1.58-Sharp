using BitNetSharp.App;
using BitNetSharp.Core;

namespace BitNetSharp.Tests;

[CollectionDefinition("Benchmark environment", DisableParallelization = true)]
public sealed class BenchmarkEnvironmentCollectionDefinition
{
}

[Collection("Benchmark environment")]
[Trait(TestCategories.Category, TestCategories.SlowLane)]
public sealed class HostedAgentBenchmarksExecutionTests
{
    private const string BlankSeparatorLine = " ";

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
        var validationSamples = BenchmarkFixtureTestData.CreateCompactWikiText2ValidationSamples();
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
    public void WikiText2TokenizedSplitsLoadFromRepositoryLocalFixtures()
    {
        var trainingSamples = BitNetBenchmarkFixtures.WikiText2TrainingSamples;
        var validationSamples = BitNetBenchmarkFixtures.WikiText2ValidationSamples;
        var testSamples = BitNetBenchmarkFixtures.WikiText2TestSamples;

        Assert.Equal(36718, trainingSamples.Count);
        Assert.Equal(3760, validationSamples.Count);
        Assert.Equal(4358, testSamples.Count);

        Assert.Equal(BlankSeparatorLine, trainingSamples[0]);
        Assert.Equal(" = Valkyria Chronicles III = ", trainingSamples[1]);
        Assert.Equal(BlankSeparatorLine, trainingSamples[2]);

        Assert.Equal(BlankSeparatorLine, validationSamples[0]);
        Assert.Equal(" = Homarus gammarus = ", validationSamples[1]);
        Assert.Equal(BlankSeparatorLine, validationSamples[2]);

        Assert.Equal(BlankSeparatorLine, testSamples[0]);
        Assert.Equal(" = Robert <unk> = ", testSamples[1]);
        Assert.Equal(BlankSeparatorLine, testSamples[2]);

        Assert.Contains(validationSamples, static sample => sample.Contains("first New Zealand side to perform a <unk>", StringComparison.Ordinal));
        Assert.StartsWith(" Common starlings are trapped for food in some Mediterranean countries .", trainingSamples[^2], StringComparison.Ordinal);
        Assert.EndsWith("it may still be seen as an acquired taste . ", trainingSamples[^2], StringComparison.Ordinal);
        Assert.StartsWith(" The <unk> is credited with sparking a resurgence in the popularity of pool in the United States", testSamples[^2], StringComparison.Ordinal);
        Assert.Contains("Minnesota <unk>", testSamples[^2], StringComparison.Ordinal);

        Assert.Equal(BlankSeparatorLine, trainingSamples[^1]);
        Assert.Equal(" = = = Television roles = = = ", validationSamples[^3]);
        Assert.Equal(BlankSeparatorLine, validationSamples[^2]);
        Assert.Equal(BlankSeparatorLine, validationSamples[^1]);
        Assert.Equal(BlankSeparatorLine, testSamples[^1]);
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
