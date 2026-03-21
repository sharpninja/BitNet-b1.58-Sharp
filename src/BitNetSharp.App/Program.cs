using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Quantization;
using Microsoft.Extensions.DependencyInjection;

// Twenty columns keeps the console histogram readable without wrapping typical terminals.
const int DefaultHistogramWidth = 20;

var command = args.FirstOrDefault()?.ToLowerInvariant() ?? "chat";
var verbosity = ParseVerbosity(args);
var modelSpecifier = ParseOption(args, "--model=") ?? HostedAgentModelFactory.DefaultModelId;
var enableBucketing = args.Any(a => string.Equals(a, "--enable-bucketing", StringComparison.OrdinalIgnoreCase));

if (command == "benchmark")
{
    HostedAgentBenchmarkRunner.Run(HostedAgentBenchmarkOptions.Parse(args, verbosity));
    return;
}

if (command == "benchmark-report")
{
    var reportDirectory = ParseOption(args, "--output=");
    var commitHash = ParseOption(args, "--commit=");
    var outputPath = await HostedAgentBenchmarkReportRunner.RunAsync(
        HostedAgentBenchmarkOptions.Parse(args, verbosity),
        reportDirectory,
        commitHash);
    Console.WriteLine($"Saved benchmark comparison report to {outputPath}");
    return;
}

if (command == "datagen")
{
    var outputPath = await DataGenCommand.RunAsync(args, verbosity);
    Console.WriteLine($"Saved synthetic dataset to {outputPath}");
    return;
}

using var model = HostedAgentModelFactory.Create(modelSpecifier, verbosity, enableChainBuckets: enableBucketing, enableSequenceCompression: enableBucketing);

// When --enable-bucketing is requested for the built-in BitNet model, mine chain buckets
// from the default training corpus and attach them so speculative decoding and sequence
// compression are active for the current session.
if (enableBucketing && model is BitNetHostedAgentModel bitNetBucketingModel)
{
    var bucketCorpus = BitNetTrainingCorpus.CreateDefaultExamples();
    var bucketTable = bitNetBucketingModel.Model.MineAndLoadBuckets(bucketCorpus);

    if (verbosity != VerbosityLevel.Quiet)
    {
        Console.WriteLine($"Bucketing active: {bucketTable.Count} chain bucket(s) mined from default training corpus.");
    }
}

using var host = BitNetAgentHost.Build(model);
var hostSummary = host.Services.GetRequiredService<BitNetHostSummary>();

switch (command)
{
    case "train":
        if (model is ITrainableHostedAgentModel trainableModel)
        {
            var examples = BitNetTrainingCorpus.CreateDefaultExamples();
            var trainingEpochs = GetTrainingEpochs(model);
            trainableModel.Train(examples, trainingEpochs);
            Console.WriteLine($"Trained '{model.ModelId}' on {examples.Count} default examples for {trainingEpochs} epochs.");
        }
        else
        {
            Console.WriteLine($"Model '{model.ModelId}' does not expose repository-local training.");
        }

        Console.WriteLine(FormatModelSummary(model));
        break;

    case "visualize":
        Console.WriteLine(FormatModelSummary(model));
        Console.WriteLine();
        if (model is IInspectableHostedAgentModel inspectableModel)
        {
            Console.WriteLine(FormatWeightHistogram(inspectableModel.GetTernaryWeightStats()));
        }
        else
        {
            Console.WriteLine($"Model '{model.ModelId}' does not expose repository weight-sign inspection.");
        }
        break;

    case "paper-audit":
        if (model is BitNetHostedAgentModel bitNetModel)
        {
            Console.WriteLine(BitNetPaperAuditCommand.FormatReport(BitNetPaperAuditor.CreateReport(bitNetModel.Model)));
        }
        else
        {
            Console.WriteLine($"Model '{model.ModelId}' does not expose the paper-aligned BitNet audit surface.");
        }

        break;

    case "host":
        Console.WriteLine($"Agent: {hostSummary.AgentName}");
        Console.WriteLine($"Model: {hostSummary.ModelId}");
        Console.WriteLine($"Display: {hostSummary.DisplayName}");
        Console.WriteLine($"Hosting: {hostSummary.HostingFramework}");
        Console.WriteLine($"Language: {hostSummary.PrimaryLanguage}");
        Console.WriteLine($"Verbosity: {hostSummary.Verbosity}");
        break;

    case "chat":
    default:
        var prompt = ParsePrompt(args);
        var result = await model.GetResponseAsync(prompt, ParseMaxTokens(args));
        Console.WriteLine(result.Text);

        if (verbosity != VerbosityLevel.Quiet)
        {
            foreach (var line in result.Diagnostics)
            {
                Console.WriteLine(line);
            }
        }

        break;
}

static string FormatModelSummary(IHostedAgentModel model) => string.Join(Environment.NewLine, model.DescribeModel());

static string FormatWeightHistogram(TernaryWeightStats stats)
{
    var max = Math.Max(stats.NegativeCount, Math.Max(stats.ZeroCount, stats.PositiveCount));
    max = Math.Max(max, 1);
    var scale = DefaultHistogramWidth / (double)max;

    return string.Join(
        Environment.NewLine,
        [
            "Weight sign distribution",
            FormatBar("-1", stats.NegativeCount, max, scale),
            FormatBar(" 0", stats.ZeroCount, max, scale),
            FormatBar("+1", stats.PositiveCount, max, scale)
        ]);
}

static string FormatBar(string label, int value, int max, double scale)
{
    if (max <= 0)
    {
        return $"{label}:  {value}";
    }

    var width = Math.Max(0, (int)Math.Round(value * scale));
    return $"{label}: {new string('#', width)} {value}";
}

static VerbosityLevel ParseVerbosity(string[] args)
{
    return ParseOption(args, "--verbosity=")?.ToLowerInvariant() switch
    {
        "quiet" => VerbosityLevel.Quiet,
        "verbose" => VerbosityLevel.Verbose,
        _ => VerbosityLevel.Normal
    };
}

static string ParsePrompt(string[] args)
{
    var positional = args
        .Skip(1)
        .Where(argument => !argument.StartsWith("--", StringComparison.Ordinal))
        .ToArray();

    return positional.Length == 0 ? "hello" : string.Join(' ', positional);
}

static int? ParseMaxTokens(string[] args)
{
    var value = ParseOption(args, "--max-tokens=");
    return int.TryParse(value, out var parsed) ? parsed : null;
}

static int GetTrainingEpochs(IHostedAgentModel model) =>
    string.Equals(model.ModelId, HostedAgentModelFactory.TraditionalLocalModelId, StringComparison.Ordinal)
        ? TraditionalLocalModel.DefaultTrainingEpochs
        : 3;

static string? ParseOption(IEnumerable<string> args, string prefix) =>
    args.FirstOrDefault(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        ?.Split('=', 2)
        .LastOrDefault();
