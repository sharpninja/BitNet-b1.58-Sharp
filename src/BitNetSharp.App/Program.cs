using BitNetSharp.App;
using BitNetSharp.Core;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

var command = args.FirstOrDefault()?.ToLowerInvariant() ?? "chat";
var prompt = args.Length > 1 ? string.Join(' ', args.Skip(1)) : "hello";
var verbosity = ParseVerbosity(args);

var (model, trainingReport) = BitNetBootstrap.CreateSeededModel(verbosity);
using var host = BitNetAgentHost.Build(model);
var visualizer = new BitNetVisualizer();
var hostSummary = host.Services.GetRequiredService<BitNetHostSummary>();

switch (command)
{
    case "train":
        Console.WriteLine($"Samples seen: {trainingReport.SamplesSeen}");
        Console.WriteLine($"Average loss: {trainingReport.AverageLoss:0.000}");
        Console.WriteLine(visualizer.CreateReport(trainingReport).LossChart);
        break;

    case "visualize":
        var visualization = visualizer.CreateReport(trainingReport);
        Console.WriteLine(visualization.LossChart);
        Console.WriteLine();
        Console.WriteLine(visualization.WeightHistogram);
        Console.WriteLine();
        Console.WriteLine(visualization.Csv.TrimEnd());
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
