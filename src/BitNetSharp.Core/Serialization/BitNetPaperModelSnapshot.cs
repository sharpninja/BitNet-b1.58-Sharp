using BitNetSharp.Core.Models;

namespace BitNetSharp.Core;

internal sealed record BitNetPaperModelSnapshot(
    string ModelId,
    int BootstrapSeed,
    BitNetConfig Config,
    IReadOnlyList<string> Vocabulary,
    int MaxResponseTokens,
    string PrimaryLanguage,
    bool EnableChainBuckets,
    bool EnableSequenceCompression,
    double ChainBucketAcceptanceThreshold,
    float[,] TokenEmbeddings,
    IReadOnlyList<float[,]> TransformerProjectionWeights,
    IReadOnlyList<float[]> NormScales,
    float[,] OutputHeadWeights,
    IReadOnlyDictionary<string, int[]> MemorizedResponses)
{
    internal const int DefaultBootstrapSeed = 42;

    public static BitNetPaperModelSnapshot Capture(BitNetPaperModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        return new BitNetPaperModelSnapshot(
            model.ModelId,
            DefaultBootstrapSeed,
            model.Config,
            [.. model.Options.Vocabulary],
            model.Options.MaxResponseTokens,
            model.Options.PrimaryLanguage,
            model.Options.EnableChainBuckets,
            model.Options.EnableSequenceCompression,
            model.Options.ChainBucketAcceptanceThreshold,
            CloneMatrix(model.ExportTokenEmbeddings()),
            model.ExportTransformerProjectionWeights().Select(CloneMatrix).ToArray(),
            model.ExportNormScales().Select(CloneVector).ToArray(),
            CloneMatrix(model.ExportOutputHeadWeights()),
            CloneMemorizedResponses(model.ExportMemorizedResponses()));
    }

    public BitNetPaperModel Restore(VerbosityLevel verbosity = VerbosityLevel.Normal)
    {
        var model = new BitNetPaperModel(
            new BitNetOptions(
                [.. Vocabulary],
                verbosity,
                MaxResponseTokens,
                PrimaryLanguage,
                EnableChainBuckets,
                EnableSequenceCompression,
                ChainBucketAcceptanceThreshold),
            Config,
            BootstrapSeed);

        model.ImportTokenEmbeddings(CloneMatrix(TokenEmbeddings));
        model.ImportTransformerProjectionWeights(TransformerProjectionWeights.Select(CloneMatrix).ToArray());
        model.ImportNormScales(NormScales.Select(CloneVector).ToArray());
        model.ImportOutputHeadWeights(CloneMatrix(OutputHeadWeights));
        model.ImportMemorizedResponses(CloneMemorizedResponses(MemorizedResponses));
        return model;
    }

    internal static float[,] CloneMatrix(float[,] matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);

        var clone = new float[matrix.GetLength(0), matrix.GetLength(1)];
        Array.Copy(matrix, clone, matrix.Length);
        return clone;
    }

    internal static float[] CloneVector(IReadOnlyList<float> vector)
    {
        ArgumentNullException.ThrowIfNull(vector);

        var clone = new float[vector.Count];
        for (var index = 0; index < vector.Count; index++)
        {
            clone[index] = vector[index];
        }

        return clone;
    }

    internal static Dictionary<string, int[]> CloneMemorizedResponses(IReadOnlyDictionary<string, IReadOnlyList<int>> memorizedResponses)
    {
        ArgumentNullException.ThrowIfNull(memorizedResponses);

        return memorizedResponses.ToDictionary(
            static pair => pair.Key,
            static pair => pair.Value.ToArray(),
            StringComparer.Ordinal);
    }

    internal static Dictionary<string, int[]> CloneMemorizedResponses(IReadOnlyDictionary<string, int[]> memorizedResponses)
    {
        ArgumentNullException.ThrowIfNull(memorizedResponses);

        return memorizedResponses.ToDictionary(
            static pair => pair.Key,
            static pair => pair.Value.ToArray(),
            StringComparer.Ordinal);
    }
}
