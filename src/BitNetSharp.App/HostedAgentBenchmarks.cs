using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using BitNetSharp.Core;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace BitNetSharp.App;

public static class HostedAgentBenchmarkRunner
{
    public static void Run(HostedAgentBenchmarkOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.StoreInEnvironment();
        BenchmarkRunner.Run<HostedAgentHostBenchmarks>();
        BenchmarkRunner.Run<HostedAgentResponseBenchmarks>();
        BenchmarkRunner.Run<HostedAgentStreamingBenchmarks>();
        BenchmarkRunner.Run<HostedAgentTrainingBenchmarks>();
    }
}

internal static class HostedAgentBenchmarkModelBootstrap
{
    public static IHostedAgentModel CreatePreparedModel(
        string modelSpecifier,
        HostedAgentBenchmarkOptions options,
        IReadOnlyList<TrainingExample>? trainingExamples = null)
    {
        var model = HostedAgentModelFactory.Create(
            modelSpecifier,
            options.Verbosity,
            trainingExamples,
            enableChainBuckets: options.EnableBucketing,
            enableSequenceCompression: options.EnableBucketing);

        if (options.EnableBucketing && model is BitNetHostedAgentModel bitNetModel)
        {
            bitNetModel.Model.MineAndLoadBuckets(trainingExamples ?? BitNetTrainingCorpus.CreateDefaultExamples());
        }

        return model;
    }
}

public abstract class HostedAgentBenchmarkBase
{
    protected HostedAgentBenchmarkOptions Options => HostedAgentBenchmarkOptions.LoadFromEnvironment();

    [ParamsSource(nameof(ModelSpecifiers))]
    public string ModelSpecifier { get; set; } = HostedAgentModelFactory.DefaultModelId;

    public IEnumerable<string> ModelSpecifiers => Options.ModelSpecifiers;
}

public abstract class TrainableHostedAgentBenchmarkBase : HostedAgentBenchmarkBase
{
    [ParamsSource(nameof(TrainableModelSpecifiers))]
    public new string ModelSpecifier { get; set; } = HostedAgentModelFactory.TraditionalLocalModelId;

    public IEnumerable<string> TrainableModelSpecifiers => Options.ModelSpecifiers
        .Where(IsTrainableSpecifier)
        .DefaultIfEmpty(HostedAgentModelFactory.DefaultModelId);

    private static bool IsTrainableSpecifier(string specifier)
    {
        if (File.Exists(specifier))
        {
            return true;
        }

        return string.Equals(specifier, HostedAgentModelFactory.DefaultModelId, StringComparison.OrdinalIgnoreCase)
            || string.Equals(specifier, HostedAgentModelFactory.TraditionalLocalModelId, StringComparison.OrdinalIgnoreCase);
    }
}

[MemoryDiagnoser, ShortRunJob]
public class HostedAgentResponseBenchmarks : HostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Generate a response for a prompt")]
    public async Task<string> GenerateResponseForPrompt()
    {
        using var model = HostedAgentBenchmarkModelBootstrap.CreatePreparedModel(ModelSpecifier, Options);
        using var host = BitNetAgentHost.Build(model);
        var chatClient = host.Services.GetRequiredService<IChatClient>();
        var response = await chatClient.GetResponseAsync(
            [new ChatMessage(ChatRole.User, Options.Prompt)],
            new ChatOptions { MaxOutputTokens = Options.MaxOutputTokens });

        return response.Text;
    }
}

[MemoryDiagnoser, ShortRunJob]
public class HostedAgentStreamingBenchmarks : HostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Stream a response for a prompt")]
    public async Task<int> StreamResponseForPrompt()
    {
        using var model = HostedAgentBenchmarkModelBootstrap.CreatePreparedModel(ModelSpecifier, Options);
        using var host = BitNetAgentHost.Build(model);
        var chatClient = host.Services.GetRequiredService<IChatClient>();
        var count = 0;

        await foreach (var _ in chatClient.GetStreamingResponseAsync(
                           [new ChatMessage(ChatRole.User, Options.Prompt)],
                           new ChatOptions { MaxOutputTokens = Options.MaxOutputTokens }))
        {
            // Count each streamed update so the benchmark measures the end-to-end streaming path.
            count++;
        }

        return count;
    }
}

[MemoryDiagnoser, ShortRunJob]
public class HostedAgentTrainingBenchmarks : TrainableHostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Train the selected model on the TinyLlama-1.1B benchmark dataset")]
    public int TrainSelectedModel()
    {
        var examples = BitNetTrainingCorpus.CreateBenchmarkExamples();
        using var model = HostedAgentBenchmarkModelBootstrap.CreatePreparedModel(ModelSpecifier, Options, examples);
        if (model is not ITrainableHostedAgentModel trainableModel)
        {
            return 0;
        }

        var epochs = string.Equals(ModelSpecifier, HostedAgentModelFactory.TraditionalLocalModelId, StringComparison.Ordinal)
            ? TraditionalLocalModel.DefaultTrainingEpochs
            : 3;
        trainableModel.Train(examples, epochs);
        return examples.Count;
    }
}

[MemoryDiagnoser, ShortRunJob]
public class HostedAgentHostBenchmarks : HostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Build the agent host for the selected model")]
    public string BuildAgentHost()
    {
        using var model = HostedAgentBenchmarkModelBootstrap.CreatePreparedModel(ModelSpecifier, Options);
        using var host = BitNetAgentHost.Build(model);
        var summary = host.Services.GetRequiredService<BitNetHostSummary>();
        return summary.ModelId;
    }
}
