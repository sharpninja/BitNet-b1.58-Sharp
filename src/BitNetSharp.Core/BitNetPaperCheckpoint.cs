using System.Text.Json;
using BitNetSharp.Core.Bucketing;
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
    float[][]? TokenEmbeddings,
    float[][][]? TransformerProjectionWeights,
    float[][]? NormScales,
    Dictionary<string, int[]>? MemorizedResponses,
    int MaxResponseTokens,
    string PrimaryLanguage,
    bool EnableChainBuckets,
    bool EnableSequenceCompression,
    double ChainBucketAcceptanceThreshold);

public static class BitNetPaperCheckpoint
{
    private const string FormatName = "bitnet-b1.58-sharp.repository-checkpoint.v1";
    private const string BucketSidecarFileName = "chain-buckets.bin";

    public static void Save(BitNetPaperModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var snapshot = BitNetPaperModelSnapshot.Capture(model);
        var document = new BitNetPaperCheckpointDocument(
            FormatName,
            snapshot.ModelId,
            snapshot.BootstrapSeed,
            snapshot.Config,
            [.. snapshot.Vocabulary],
            ToJagged(snapshot.OutputHeadWeights),
            ToJagged(snapshot.TokenEmbeddings),
            snapshot.TransformerProjectionWeights.Select(ToJagged).ToArray(),
            snapshot.NormScales.Select(static scale => scale.ToArray()).ToArray(),
            BitNetPaperModelSnapshot.CloneMemorizedResponses(snapshot.MemorizedResponses),
            snapshot.MaxResponseTokens,
            snapshot.PrimaryLanguage,
            snapshot.EnableChainBuckets,
            snapshot.EnableSequenceCompression,
            snapshot.ChainBucketAcceptanceThreshold);
        File.WriteAllText(path, JsonSerializer.Serialize(document, new JsonSerializerOptions { WriteIndented = true }));
        SaveBucketSidecar(model.BucketTable, GetBucketSidecarPath(path));
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

        var acceptanceThreshold = document.ChainBucketAcceptanceThreshold > 0d
            ? document.ChainBucketAcceptanceThreshold
            : 0.85d;
        var baselineModel = new BitNetPaperModel(
            new BitNetOptions(
                document.Vocabulary.ToArray(),
                verbosity,
                document.MaxResponseTokens,
                document.PrimaryLanguage,
                document.EnableChainBuckets,
                document.EnableSequenceCompression,
                acceptanceThreshold),
            document.Config,
            document.BootstrapSeed);
        var baselineSnapshot = BitNetPaperModelSnapshot.Capture(baselineModel);
        var snapshot = baselineSnapshot with
        {
            ModelId = document.ModelId,
            BootstrapSeed = document.BootstrapSeed,
            Config = document.Config,
            Vocabulary = document.Vocabulary.ToArray(),
            MaxResponseTokens = document.MaxResponseTokens,
            PrimaryLanguage = document.PrimaryLanguage,
            EnableChainBuckets = document.EnableChainBuckets,
            EnableSequenceCompression = document.EnableSequenceCompression,
            ChainBucketAcceptanceThreshold = acceptanceThreshold,
            TokenEmbeddings = document.TokenEmbeddings is null ? baselineSnapshot.TokenEmbeddings : ToMatrix(document.TokenEmbeddings),
            TransformerProjectionWeights = document.TransformerProjectionWeights is null
                ? baselineSnapshot.TransformerProjectionWeights
                : document.TransformerProjectionWeights.Select(ToMatrix).ToArray(),
            NormScales = document.NormScales is null
                ? baselineSnapshot.NormScales
                : document.NormScales.Select(BitNetPaperModelSnapshot.CloneVector).ToArray(),
            OutputHeadWeights = ToMatrix(document.OutputHeadWeights),
            MemorizedResponses = document.MemorizedResponses ?? new Dictionary<string, int[]>(StringComparer.Ordinal)
        };
        var model = snapshot.Restore(verbosity);

        var bucketSidecarPath = GetBucketSidecarPath(path);
        if ((document.EnableChainBuckets || document.EnableSequenceCompression) && File.Exists(bucketSidecarPath))
        {
            model.LoadBucketTable(LoadBucketSidecar(bucketSidecarPath));
        }

        return model;
    }

    public static BitNetPaperCheckpointValidationResult ValidateRoundTrip(BitNetPaperModel model, string prompt)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);

        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-paper-checkpoint-{Guid.NewGuid():N}");
        var checkpointPath = Path.Combine(tempDirectory, "model.bitnet.json");
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
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    private static string GetBucketSidecarPath(string checkpointPath)
    {
        var directory = Path.GetDirectoryName(checkpointPath);
        return string.IsNullOrWhiteSpace(directory)
            ? BucketSidecarFileName
            : Path.Combine(directory, BucketSidecarFileName);
    }

    private static void SaveBucketSidecar(ChainBucketTable? bucketTable, string path)
    {
        if (bucketTable is null || bucketTable.Count == 0)
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }

            return;
        }

        ChainBucketTableBinarySerializer.Save(bucketTable, path);
    }

    private static ChainBucketTable LoadBucketSidecar(string path) => ChainBucketTableBinarySerializer.Load(path);

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
