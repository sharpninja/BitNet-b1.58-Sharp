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

                Assert.Contains("Top next-token predictions:", response, StringComparison.Ordinal);
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
    public async Task TrainingBenchmarkRunsTheSharedDefaultDatasetForTrainableModels()
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

                Assert.Equal(BitNetTrainingCorpus.CreateDefaultExamples().Count, trainedExamples);
                return Task.CompletedTask;
            });
    }

    [Fact]
    public async Task TrainingBenchmarkRunsThePaperAlignedTrainingPath()
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

                Assert.Equal(BitNetTrainingCorpus.CreateDefaultExamples().Count, trainedExamples);
                return Task.CompletedTask;
            });
    }

    [Fact]
    public void PerplexityEvaluationProducesFiniteValuesForBuiltInModels()
    {
        var bitNetModel = BitNetPaperModel.CreateDefault();
        var traditionalModel = TraditionalLocalModel.CreateDefault();

        var bitNetPerplexity = bitNetModel.CalculatePerplexity(BitNetBenchmarkFixtures.WikiText2ValidationSamples);
        var traditionalPerplexity = traditionalModel.CalculatePerplexity(BitNetBenchmarkFixtures.WikiText2ValidationSamples);

        Assert.True(double.IsFinite(bitNetPerplexity));
        Assert.True(double.IsFinite(traditionalPerplexity));
        Assert.True(bitNetPerplexity > 0d);
        Assert.True(traditionalPerplexity > 0d);
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
