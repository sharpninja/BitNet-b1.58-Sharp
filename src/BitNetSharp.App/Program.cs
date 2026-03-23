using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Training;
using BitNetSharp.Core.Quantization;
using Microsoft.Extensions.DependencyInjection;

// Twenty columns keeps the console histogram readable without wrapping typical terminals.
const int DefaultHistogramWidth = 20;

var command = args.FirstOrDefault()?.ToLowerInvariant() ?? "chat";
var verbosity = ParseVerbosity(args);
var modelSpecifier = ParseOption(args, "--model=") ?? ParseOption(args, "--input=") ?? HostedAgentModelFactory.DefaultModelId;
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

if (command == "train")
{
    await TrainingCommand.RunAsync(
        args.Skip(1).ToArray(),
        async (options, cancellationToken) =>
        {
            var trainingDataset = TrainingDatasetLoader.Load(options.Dataset);
            var validationDataset = string.IsNullOrWhiteSpace(options.EvaluationDataset)
                ? null
                : TrainingDatasetLoader.Load(options.EvaluationDataset);

            using var trainingModel = HostedAgentModelFactory.Create(
                modelSpecifier,
                verbosity,
                trainingDataset.Examples,
                enableChainBuckets: enableBucketing,
                enableSequenceCompression: enableBucketing);

            if (enableBucketing && trainingModel is BitNetHostedAgentModel bucketedBitNetModel)
            {
                bucketedBitNetModel.Model.MineAndLoadBuckets(trainingDataset.Examples);
            }

            var report = TrainSelectedModel(trainingModel, trainingDataset, validationDataset, options);
            var cadenceAlreadySavedFinalCheckpoint = options.CheckpointEvery is int checkpointEvery
                && checkpointEvery > 0
                && options.Epochs % checkpointEvery == 0
                && report.Checkpoints?.Any(checkpoint => checkpoint.Epoch == report.Epochs) == true;
            if (options.SaveCheckpoint && !cadenceAlreadySavedFinalCheckpoint)
            {
                var checkpointPath = SaveCheckpoint(trainingModel, options, report.Epochs);
                report = report with
                {
                    Checkpoints =
                    [
                        .. report.Checkpoints ?? [],
                        new TrainingCheckpointSummary(report.Epochs, report.SamplesSeen, checkpointPath)
                    ]
                };
            }

            cancellationToken.ThrowIfCancellationRequested();
            return report;
        });
    return;
}

if (command == "export")
{
    var outputPath = ParseOption(args, "--output=");
    if (string.IsNullOrWhiteSpace(outputPath))
    {
        throw new ArgumentException("The export command requires --output=<path.gguf>.");
    }

    using var exportModel = HostedAgentModelFactory.Create(
        modelSpecifier,
        verbosity,
        enableChainBuckets: enableBucketing,
        enableSequenceCompression: enableBucketing);

    if (enableBucketing && exportModel is BitNetHostedAgentModel bucketedExportModel && bucketedExportModel.Model.BucketTable is null)
    {
        bucketedExportModel.Model.MineAndLoadBuckets(BitNetTrainingCorpus.CreateDefaultExamples());
    }

    if (exportModel is not BitNetHostedAgentModel bitNetExportModel)
    {
        throw new InvalidOperationException($"Model '{exportModel.ModelId}' does not expose BitNet GGUF export.");
    }

    BitNetPaperGguf.Save(bitNetExportModel.Model, outputPath);
    Console.WriteLine($"Saved GGUF model to {Path.GetFullPath(outputPath)}");
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

static TrainingReport TrainSelectedModel(
    IHostedAgentModel model,
    TrainingDataset trainingDataset,
    TrainingDataset? validationDataset,
    TrainingCommandOptions options)
{
    ArgumentNullException.ThrowIfNull(model);
    ArgumentNullException.ThrowIfNull(trainingDataset);
    ArgumentNullException.ThrowIfNull(options);

    var report = model switch
    {
        BitNetHostedAgentModel bitNetModel => bitNetModel.Model.Train(
            trainingDataset.Examples,
            new BitNetTrainingOptions(
                epochs: options.Epochs,
                evaluationInterval: options.EvaluateEvery ?? 0,
                checkpointInterval: options.CheckpointEvery ?? 0,
                dataLoaderOptions: new BitNetDataLoaderOptions(
                    sequenceLength: Math.Min(bitNetModel.Model.Config.MaxSequenceLength - 1, 64),
                    batchSize: 4,
                    validationFraction: validationDataset is null ? 0.1d : 0d,
                    dropLast: false),
                compactEvaluation: options.CompactEvaluation,
                trainingDatasetName: trainingDataset.Name,
                validationDatasetName: validationDataset?.Name,
                checkpointDirectory: options.CheckpointDirectory,
                checkpointPrefix: options.CheckpointPrefix,
                externalEvaluation: validationDataset is null
                    ? null
                    : _ => CreateValidationSummary(bitNetModel, validationDataset))),
        TraditionalLocalHostedAgentModel traditionalModel => traditionalModel.Train(
            trainingDataset.Examples,
            Math.Max(TraditionalLocalModel.DefaultTrainingEpochs, options.Epochs)),
        ITrainableHostedAgentModel trainableModel => trainableModel.Train(trainingDataset.Examples, options.Epochs),
        _ => throw new InvalidOperationException($"Model '{model.ModelId}' does not expose repository-local training.")
    };

    if (validationDataset is null)
    {
        return report with { TrainingDataset = trainingDataset.Name };
    }

    var shouldAppendFinalValidationSummary = options.EvaluateEvery is not int evaluateEvery
        || evaluateEvery <= 0
        || options.Epochs % evaluateEvery != 0;

    return report with
    {
        TrainingDataset = trainingDataset.Name,
        ValidationDataset = validationDataset.Name,
        EvaluationSummaries = shouldAppendFinalValidationSummary
            ? [.. report.EvaluationSummaries ?? [], CreateValidationSummary(model, validationDataset)]
            : report.EvaluationSummaries
    };
}

static TrainingEvaluationSummary CreateValidationSummary(IHostedAgentModel model, TrainingDataset validationDataset)
{
    var samples = validationDataset.Examples
        .Select(static example => $"{example.Prompt} {example.Response}")
        .ToArray();
    var perplexity = model switch
    {
        BitNetHostedAgentModel bitNetModel => bitNetModel.Model.CalculatePerplexity(samples),
        TraditionalLocalHostedAgentModel traditionalModel => traditionalModel.Model.CalculatePerplexity(samples),
        _ => 0d
    };

    return new TrainingEvaluationSummary(
        validationDataset.Name,
        samples.Length,
        perplexity <= 0d ? 0d : Math.Log(perplexity),
        perplexity);
}

static string SaveCheckpoint(IHostedAgentModel model, TrainingCommandOptions options, int epoch)
{
    var directory = string.IsNullOrWhiteSpace(options.CheckpointDirectory)
        ? Environment.CurrentDirectory
        : options.CheckpointDirectory!;
    Directory.CreateDirectory(directory);
    var extension = model switch
    {
        BitNetHostedAgentModel => ".bitnet.json",
        TraditionalLocalHostedAgentModel => ".traditional.json",
        _ => ".checkpoint.json"
    };
    var checkpointPath = Path.Combine(directory, $"{options.CheckpointPrefix}-epoch{epoch}{extension}");

    switch (model)
    {
        case BitNetHostedAgentModel bitNetModel:
            BitNetPaperCheckpoint.Save(bitNetModel.Model, checkpointPath);
            break;
        case TraditionalLocalHostedAgentModel traditionalModel:
            TraditionalLocalCheckpoint.Save(traditionalModel.Model, checkpointPath);
            break;
        default:
            throw new InvalidOperationException($"Model '{model.ModelId}' does not expose checkpoint persistence.");
    }

    return checkpointPath;
}

static string? ParseOption(IEnumerable<string> args, string prefix) =>
    args.FirstOrDefault(argument => argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        ?.Split('=', 2)
        .LastOrDefault();
