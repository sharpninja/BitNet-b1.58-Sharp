using BitNetSharp.Core;

namespace BitNetSharp.App;

public enum TrainingCommandReportFormat
{
    PlainText,
    Markdown,
    Json
}

public sealed record TrainingCommandOptions(
    string Dataset,
    int Epochs,
    int? EvaluateEvery,
    string? EvaluationDataset,
    int? CheckpointEvery,
    string? CheckpointDirectory,
    string? CheckpointPrefix,
    string? ReportPath,
    TrainingCommandReportFormat ReportFormat,
    bool CompactEvaluation,
    bool SaveCheckpoint,
    bool DryRun,
    bool HelpRequested)
{
    public const string DefaultDataset = "default";
    public const int DefaultEpochs = 3;
    public const string DefaultCheckpointPrefix = "training-checkpoint";

    public bool HasEvaluationSchedule => EvaluateEvery.HasValue || !string.IsNullOrWhiteSpace(EvaluationDataset);

    public bool HasCheckpointSchedule => SaveCheckpoint;

    public bool HasReportPath => !string.IsNullOrWhiteSpace(ReportPath);

    public static TrainingCommandOptions Parse(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);

        var dataset = ReadOptionalOption(args, "--dataset") ?? DefaultDataset;
        if (string.IsNullOrWhiteSpace(dataset))
        {
            throw new ArgumentException("The --dataset option requires a non-empty value.", nameof(args));
        }

        var epochs = ParsePositiveInt(ReadOptionalOption(args, "--epochs"), DefaultEpochs, "--epochs");
        var evaluateEvery = ParsePositiveNullableInt(ReadOptionalOption(args, "--eval-every"), "--eval-every");
        var evaluationDataset = ReadOptionalOption(args, "--validation-dataset", "--eval-dataset");
        var checkpointEvery = ParsePositiveNullableInt(ReadOptionalOption(args, "--checkpoint-every"), "--checkpoint-every");
        var checkpointDirectory = NormalizePathOption(ReadOptionalOption(args, "--checkpoint-dir"));
        var checkpointPrefix = ReadOptionalOption(args, "--checkpoint-prefix") ?? DefaultCheckpointPrefix;
        if (string.IsNullOrWhiteSpace(checkpointPrefix))
        {
            throw new ArgumentException("The --checkpoint-prefix option requires a non-empty value.", nameof(args));
        }

        var reportPath = NormalizePathOption(ReadOptionalOption(args, "--report-path"));
        var explicitReportFormat = ReadOptionalOption(args, "--report-format");
        var reportFormat = ParseReportFormat(
            explicitReportFormat ?? InferReportFormat(reportPath));

        var compactEvaluation = ReadSwitchState(args, "--compact-eval", "--full-eval", defaultValue: true);
        compactEvaluation = ReadSwitchState(args, "--compact-eval", "--no-compact-eval", compactEvaluation);

        var saveCheckpoint = ReadSwitchState(
            args,
            "--save-checkpoint",
            "--no-save-checkpoint",
            defaultValue: checkpointEvery.HasValue
                || checkpointDirectory is not null
                || !string.Equals(checkpointPrefix, DefaultCheckpointPrefix, StringComparison.Ordinal));
        var dryRun = HasFlag(args, "--dry-run");
        var helpRequested = HasFlag(args, "--help") || HasFlag(args, "-h");

        return new TrainingCommandOptions(
            dataset.Trim(),
            epochs,
            evaluateEvery,
            string.IsNullOrWhiteSpace(evaluationDataset) ? null : evaluationDataset.Trim(),
            checkpointEvery,
            checkpointDirectory,
            checkpointPrefix.Trim(),
            reportPath,
            reportFormat,
            compactEvaluation,
            saveCheckpoint,
            dryRun,
            helpRequested);
    }

    private static string? ReadOptionalOption(string[] args, params string[] optionNames)
    {
        foreach (var optionName in optionNames)
        {
            var value = ReadOptionalOption(args, optionName);
            if (value is not null)
            {
                return value;
            }
        }

        return null;
    }

    private static string? ReadOptionalOption(string[] args, string optionName)
    {
        var equalsPrefix = $"{optionName}=";
        var missingValueDetected = false;

        for (var index = 0; index < args.Length; index++)
        {
            var argument = args[index];
            if (argument.StartsWith(equalsPrefix, StringComparison.OrdinalIgnoreCase))
            {
                return argument[equalsPrefix.Length..];
            }

            if (string.Equals(argument, optionName, StringComparison.OrdinalIgnoreCase))
            {
                var nextIndex = index + 1;
                if (nextIndex >= args.Length)
                {
                    missingValueDetected = true;
                    continue;
                }

                var nextArgument = args[nextIndex];
                if (nextArgument.StartsWith("--", StringComparison.Ordinal))
                {
                    missingValueDetected = true;
                    continue;
                }

                return nextArgument;
            }
        }

        if (missingValueDetected)
        {
            throw new ArgumentException($"Option '{optionName}' requires a value.", nameof(args));
        }

        return null;
    }

    private static bool HasFlag(string[] args, string optionName) =>
        args.Any(argument => string.Equals(argument, optionName, StringComparison.OrdinalIgnoreCase));

    private static bool ReadSwitchState(string[] args, string positiveOption, string negativeOption, bool defaultValue)
    {
        var result = defaultValue;
        foreach (var argument in args)
        {
            if (string.Equals(argument, positiveOption, StringComparison.OrdinalIgnoreCase))
            {
                result = true;
            }
            else if (string.Equals(argument, negativeOption, StringComparison.OrdinalIgnoreCase))
            {
                result = false;
            }
        }

        return result;
    }

    private static int ParsePositiveInt(string? value, int defaultValue, string optionName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return defaultValue;
        }

        if (int.TryParse(value, out var parsed) && parsed > 0)
        {
            return parsed;
        }

        throw new ArgumentOutOfRangeException(nameof(value), $"The {optionName} option must be a positive integer.");
    }

    private static int? ParsePositiveNullableInt(string? value, string optionName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        if (int.TryParse(value, out var parsed) && parsed > 0)
        {
            return parsed;
        }

        throw new ArgumentOutOfRangeException(nameof(value), $"The {optionName} option must be a positive integer.");
    }

    private static string? NormalizePathOption(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : Path.GetFullPath(value);

    private static TrainingCommandReportFormat ParseReportFormat(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return TrainingCommandReportFormat.PlainText;
        }

        return value.Trim().ToLowerInvariant() switch
        {
            "text" or "plain" or "plaintext" => TrainingCommandReportFormat.PlainText,
            "markdown" or "md" => TrainingCommandReportFormat.Markdown,
            "json" => TrainingCommandReportFormat.Json,
            _ => throw new ArgumentException($"Unsupported report format '{value}'. Expected text, markdown, or json.", nameof(value))
        };
    }

    private static string? InferReportFormat(string? reportPath)
    {
        if (string.IsNullOrWhiteSpace(reportPath))
        {
            return null;
        }

        return Path.GetExtension(reportPath).ToLowerInvariant() switch
        {
            ".md" or ".markdown" => "markdown",
            ".json" => "json",
            _ => "text"
        };
    }
}
