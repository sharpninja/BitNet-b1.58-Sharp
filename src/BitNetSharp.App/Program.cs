using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

var command = args.FirstOrDefault()?.ToLowerInvariant() ?? "chat";
var prompt = args.Length > 1 ? string.Join(' ', args.Skip(1)) : "hello";
var verbosity = ParseVerbosity(args);

var model = BitNetBootstrap.CreatePaperModel(verbosity);
using var host = BitNetAgentHost.Build(model);
var hostSummary = host.Services.GetRequiredService<BitNetHostSummary>();

switch (command)
{
    case "train":
        Console.WriteLine("Paper-aligned transformer training is not implemented yet in this branch.");
        Console.WriteLine(FormatModelSummary(model));
        break;

    case "visualize":
        Console.WriteLine(FormatModelSummary(model));
        Console.WriteLine();
        Console.WriteLine(FormatWeightHistogram(model.GetTernaryWeightStats()));
        break;

    case "host":
        Console.WriteLine($"Agent: {hostSummary.AgentName}");
        Console.WriteLine($"Hosting: {hostSummary.HostingFramework}");
        Console.WriteLine($"Language: {hostSummary.PrimaryLanguage}");
        Console.WriteLine($"Verbosity: {hostSummary.Verbosity}");
        break;

    case "chat":
    default:
        var chatClient = host.Services.GetRequiredService<IChatClient>();
        var response = await chatClient.GetResponseAsync(
            [new ChatMessage(ChatRole.User, prompt)],
            new ChatOptions { MaxOutputTokens = model.Options.MaxResponseTokens });

        Console.WriteLine(response.Text);

        if (verbosity != VerbosityLevel.Quiet)
        {
            foreach (var line in model.GenerateResponse(prompt).Diagnostics)
            {
                Console.WriteLine(line);
            }
        }

        break;
}

static string FormatModelSummary(BitNetPaperModel model) =>
    string.Join(
        Environment.NewLine,
        [
            "Paper-aligned BitNet b1.58 transformer",
            $"Vocabulary size: {model.Config.VocabSize}",
            $"Layers: {model.Config.LayerCount}",
            $"Dimension: {model.Config.Dimension}",
            $"Hidden dimension: {model.Config.HiddenDimension}",
            $"Heads: {model.Config.HeadCount}",
            $"Max sequence length: {model.Config.MaxSequenceLength}"
        ]);

static string FormatWeightHistogram(TernaryWeightStats stats)
{
    var max = Math.Max(stats.NegativeCount, Math.Max(stats.ZeroCount, stats.PositiveCount));
    max = Math.Max(max, 1);

    return string.Join(
        Environment.NewLine,
        [
            "Ternary weight distribution",
            FormatBar("-1", stats.NegativeCount, max),
            FormatBar(" 0", stats.ZeroCount, max),
            FormatBar("+1", stats.PositiveCount, max)
        ]);
}

static string FormatBar(string label, int value, int max)
{
    const int HistogramMaxBarWidth = 20;
    var width = Math.Max(1, (int)Math.Round(value / (double)max * HistogramMaxBarWidth));
    return $"{label}: {new string('#', width)} {value}";
}

static VerbosityLevel ParseVerbosity(string[] args)
{
    var value = args
        .Select(argument => argument.ToLowerInvariant())
        .FirstOrDefault(argument => argument.StartsWith("--verbosity="));

    return value?.Split('=', 2).LastOrDefault() switch
    {
        "quiet" => VerbosityLevel.Quiet,
        "verbose" => VerbosityLevel.Verbose,
        _ => VerbosityLevel.Normal
    };
}
