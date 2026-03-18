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
        BenchmarkSwitcher.FromAssembly(typeof(HostedAgentBenchmarkRunner).Assembly).Run([]);
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
    public new IEnumerable<string> ModelSpecifiers => Options.ModelSpecifiers
        .Where(static specifier =>
            string.Equals(specifier, HostedAgentModelFactory.TraditionalLocalModelId, StringComparison.OrdinalIgnoreCase)
            || File.Exists(specifier))
        .DefaultIfEmpty(HostedAgentModelFactory.TraditionalLocalModelId);
}

[MemoryDiagnoser, ShortRunJob]
public sealed class HostedAgentResponseBenchmarks : HostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Generate a response for a prompt")]
    public async Task<string> GenerateResponseForPrompt()
    {
        using var model = HostedAgentModelFactory.Create(ModelSpecifier, Options.Verbosity);
        using var host = BitNetAgentHost.Build(model);
        var chatClient = host.Services.GetRequiredService<IChatClient>();
        var response = await chatClient.GetResponseAsync(
            [new ChatMessage(ChatRole.User, Options.Prompt)],
            new ChatOptions { MaxOutputTokens = Options.MaxOutputTokens });

        return response.Text;
    }
}

[MemoryDiagnoser, ShortRunJob]
public sealed class HostedAgentStreamingBenchmarks : HostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Stream a response for a prompt")]
    public async Task<int> StreamResponseForPrompt()
    {
        using var model = HostedAgentModelFactory.Create(ModelSpecifier, Options.Verbosity);
        using var host = BitNetAgentHost.Build(model);
        var chatClient = host.Services.GetRequiredService<IChatClient>();
        var count = 0;

        await foreach (var _ in chatClient.GetStreamingResponseAsync(
                           [new ChatMessage(ChatRole.User, Options.Prompt)],
                           new ChatOptions { MaxOutputTokens = Options.MaxOutputTokens }))
        {
            count++;
        }

        return count;
    }
}

[MemoryDiagnoser, ShortRunJob]
public sealed class HostedAgentTrainingBenchmarks : TrainableHostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Train the selected model on the default dataset")]
    public int TrainSelectedModel()
    {
        using var model = HostedAgentModelFactory.Create(ModelSpecifier, Options.Verbosity);
        if (model is not ITrainableHostedAgentModel trainableModel)
        {
            return 0;
        }

        var examples = BitNetTrainingCorpus.CreateDefaultExamples();
        trainableModel.Train(examples, epochs: 3);
        return examples.Count;
    }
}

[MemoryDiagnoser, ShortRunJob]
public sealed class HostedAgentHostBenchmarks : HostedAgentBenchmarkBase
{
    [Benchmark(Description = "SpecFlow: Build the agent host for the selected model")]
    public string BuildAgentHost()
    {
        using var model = HostedAgentModelFactory.Create(ModelSpecifier, Options.Verbosity);
        using var host = BitNetAgentHost.Build(model);
        var summary = host.Services.GetRequiredService<BitNetHostSummary>();
        return summary.ModelId;
    }
}
