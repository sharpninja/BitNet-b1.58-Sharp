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
    public void WikiText2TokenizedSplitsLoadFromRepositoryLocalFixtures()
    {
        var trainingSamples = BitNetBenchmarkFixtures.WikiText2TrainingSamples;
        var validationSamples = BitNetBenchmarkFixtures.WikiText2ValidationSamples;
        var testSamples = BitNetBenchmarkFixtures.WikiText2TestSamples;

        Assert.Equal(23767, trainingSamples.Count);
        Assert.Equal(2461, validationSamples.Count);
        Assert.Equal(2891, testSamples.Count);

        Assert.Equal(" = Valkyria Chronicles III = ", trainingSamples[0]);
        Assert.Equal(" = Homarus gammarus = ", validationSamples[0]);
        Assert.Equal(" = Robert <unk> = ", testSamples[0]);

        Assert.Contains("first New Zealand side to perform a <unk>", validationSamples[validationSamples.Count / 2], StringComparison.Ordinal);
        Assert.Equal(" Common starlings are trapped for food in some Mediterranean countries . The meat is tough and of low quality , so it is <unk> or made into <unk> . One recipe said it should be <unk> \" until tender , however long that may be \" . Even when correctly prepared , it may still be seen as an acquired taste . ", trainingSamples[^1]);
        Assert.Equal(" The <unk> is credited with sparking a resurgence in the popularity of pool in the United States , which had been on the decline for decades . The film also brought recognition to Willie <unk> , who , despite having won multiple world championships , was virtually unknown to the general public . Perhaps the greatest <unk> of the film 's popularity was a real @-@ life pool <unk> named Rudolf <unk> . <unk> claimed in an interview at the time of the film 's release that the character of Minnesota <unk> was based on <unk> , who at the time was known as \" New York <unk> \" . <unk> immediately adopted the Minnesota <unk> nickname and <unk> his association with the film into book and television deals and other ventures . Author Walter <unk> denied for the rest of his life that <unk> had played any role in the creation of the character . Other players would claim , with greater or lesser degrees of credibility , to have served as models for Fast Eddie , including Ronnie Allen , Ed Taylor , Ed Parker , and Eddie <unk> . ", testSamples[^1]);

        Assert.All(trainingSamples, static sample => Assert.False(string.IsNullOrWhiteSpace(sample)));
        Assert.All(validationSamples, static sample => Assert.False(string.IsNullOrWhiteSpace(sample)));
        Assert.All(testSamples, static sample => Assert.False(string.IsNullOrWhiteSpace(sample)));
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
