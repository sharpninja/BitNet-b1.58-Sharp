using System.Text.Json;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Core;

public sealed record BitNetPaperCheckpointValidationResult(
    string Prompt,
    string OriginalResponse,
    string ReloadedResponse,
    bool ResponsesMatch);

internal sealed record BitNetPaperCheckpointDocument(
    string Format,
    string ModelId,
    int BootstrapSeed,
    BitNetConfig Config,
    IReadOnlyList<string> Vocabulary,
    float[][] OutputHeadWeights,
    int MaxResponseTokens,
    string PrimaryLanguage);

public static class BitNetPaperCheckpoint
{
    private const string FormatName = "bitnet-b1.58-sharp.repository-checkpoint.v1";
    private const int BootstrapSeed = 42;

    public static void Save(BitNetPaperModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var document = new BitNetPaperCheckpointDocument(
            FormatName,
            model.ModelId,
            BootstrapSeed,
            model.Config,
            model.Options.Vocabulary.ToArray(),
            ToJagged(model.ExportOutputHeadWeights()),
            model.Options.MaxResponseTokens,
            model.Options.PrimaryLanguage);
        File.WriteAllText(path, JsonSerializer.Serialize(document, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static BitNetPaperModel Load(string path, VerbosityLevel verbosity = VerbosityLevel.Normal)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var document = JsonSerializer.Deserialize<BitNetPaperCheckpointDocument>(File.ReadAllText(path))
            ?? throw new InvalidOperationException("Could not deserialize the BitNet paper checkpoint document.");
        if (!string.Equals(document.Format, FormatName, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Unsupported checkpoint format '{document.Format}'.");
        }

        var model = new BitNetPaperModel(
            new BitNetOptions(
                document.Vocabulary.ToArray(),
                verbosity,
                document.MaxResponseTokens,
                document.PrimaryLanguage),
            document.Config,
            document.BootstrapSeed);
        model.ImportOutputHeadWeights(ToMatrix(document.OutputHeadWeights));
        return model;
    }

    public static BitNetPaperCheckpointValidationResult ValidateRoundTrip(BitNetPaperModel model, string prompt)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

        var checkpointPath = Path.Combine(Path.GetTempPath(), $"bitnet-paper-checkpoint-{Guid.NewGuid():N}.json");
        try
        {
            Save(model, checkpointPath);
            var reloaded = Load(checkpointPath, model.Options.Verbosity);
            var original = model.GenerateResponse(prompt, maxTokens: 4);
            var roundTripped = reloaded.GenerateResponse(prompt, maxTokens: 4);
            return new BitNetPaperCheckpointValidationResult(
                prompt,
                original.ResponseText,
                roundTripped.ResponseText,
                string.Equals(original.ResponseText, roundTripped.ResponseText, StringComparison.Ordinal));
        }
        finally
        {
            if (File.Exists(checkpointPath))
            {
                File.Delete(checkpointPath);
            }
        }
    }

    private static float[][] ToJagged(float[,] matrix)
    {
        var result = new float[matrix.GetLength(0)][];
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            result[row] = new float[matrix.GetLength(1)];
            for (var column = 0; column < matrix.GetLength(1); column++)
            {
                result[row][column] = matrix[row, column];
            }
        }

        return result;
    }

    private static float[,] ToMatrix(float[][] matrix)
    {
        if (matrix.Length == 0)
        {
            return new float[0, 0];
        }

        var result = new float[matrix.Length, matrix[0].Length];
        for (var row = 0; row < matrix.Length; row++)
        {
            for (var column = 0; column < matrix[row].Length; column++)
            {
                result[row, column] = matrix[row][column];
            }
        }

        return result;
    }
}
