using System.Text.Json;

namespace BitNetSharp.Core;

public sealed record TraditionalLocalCheckpointValidationResult(
    string Prompt,
    string OriginalResponse,
    string ReloadedResponse,
    bool ResponsesMatch);

internal sealed record TraditionalLocalCheckpointDocument(
    string Format,
    string ModelId,
    int Seed,
    int EmbeddingDimension,
    int ContextWindow,
    IReadOnlyList<string> Vocabulary,
    float[] TokenEmbeddings,
    float[] OutputWeights,
    float[] OutputBias,
    int MaxResponseTokens,
    string PrimaryLanguage);

public static class TraditionalLocalCheckpoint
{
    private const string FormatName = "traditional-local.repository-checkpoint.v1";

    public static void Save(TraditionalLocalModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var document = new TraditionalLocalCheckpointDocument(
            FormatName,
            model.ModelId,
            model.Seed,
            model.EmbeddingDimension,
            model.ContextWindow,
            model.Options.Vocabulary.ToArray(),
            model.ExportTokenEmbeddings(),
            model.ExportOutputWeights(),
            model.ExportOutputBias(),
            model.Options.MaxResponseTokens,
            model.Options.PrimaryLanguage);
        File.WriteAllText(path, JsonSerializer.Serialize(document, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static TraditionalLocalModel Load(string path, VerbosityLevel verbosity = VerbosityLevel.Normal)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var document = JsonSerializer.Deserialize<TraditionalLocalCheckpointDocument>(File.ReadAllText(path))
            ?? throw new InvalidOperationException("Could not deserialize the traditional local checkpoint document.");
        if (!string.Equals(document.Format, FormatName, StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Unsupported checkpoint format '{document.Format}'.");
        }

        var model = new TraditionalLocalModel(
            new BitNetOptions(
                document.Vocabulary.ToArray(),
                verbosity,
                document.MaxResponseTokens,
                document.PrimaryLanguage),
            document.EmbeddingDimension,
            document.ContextWindow,
            document.Seed);
        model.ImportState(document.TokenEmbeddings, document.OutputWeights, document.OutputBias);
        return model;
    }

    public static TraditionalLocalCheckpointValidationResult ValidateRoundTrip(TraditionalLocalModel model, string prompt)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

        var checkpointPath = Path.Combine(Path.GetTempPath(), $"traditional-local-checkpoint-{Guid.NewGuid():N}.json");
        try
        {
            Save(model, checkpointPath);
            var reloaded = Load(checkpointPath, model.Options.Verbosity);
            var original = model.GenerateResponse(prompt, maxTokens: 4);
            var roundTripped = reloaded.GenerateResponse(prompt, maxTokens: 4);
            return new TraditionalLocalCheckpointValidationResult(
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
}
