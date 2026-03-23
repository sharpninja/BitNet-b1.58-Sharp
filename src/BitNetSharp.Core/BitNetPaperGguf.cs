using System.Text.Json;
using BitNetSharp.Core.Bucketing;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Serialization.Gguf;

namespace BitNetSharp.Core;

public static class BitNetPaperGguf
{
    private const string FormatName = "bitnet-b1.58-sharp.gguf.v1";
    private const string ArchitectureName = "bitnetsharp";
    private const string TokenEmbeddingsTensorName = "token_embeddings";
    private const string OutputNormTensorName = "output_norm.weight";
    private const string OutputTensorName = "output.weight";
    private const string VocabularyMetadataKey = "bitnetsharp.vocabulary";
    private const string MemorizedResponsesMetadataKey = "bitnetsharp.memorized_responses";

    public static void Save(BitNetPaperModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var snapshot = BitNetPaperModelSnapshot.Capture(model);
        ValidateSnapshot(snapshot);

        GgufWriter.Write(path, CreateMetadata(snapshot), CreateTensors(snapshot));
        SaveBucketSidecar(model.BucketTable, GetBucketSidecarPath(path));
    }

    public static BitNetPaperModel Load(string path, VerbosityLevel verbosity = VerbosityLevel.Normal)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var document = GgufReader.Read(path);
        ValidateMetadata(document.Metadata);

        var config = ReadConfig(document.Metadata);
        var vocabulary = DeserializeVocabulary(GetRequiredString(document.Metadata, VocabularyMetadataKey));

        var expectedTensorNames = CreateExpectedTensorNames(config);
        var tensors = document.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        var missingTensorNames = expectedTensorNames.Where(name => !tensors.ContainsKey(name)).ToArray();
        var unexpectedTensorNames = tensors.Keys.Where(name => !expectedTensorNames.Contains(name, StringComparer.Ordinal)).ToArray();
        if (missingTensorNames.Length > 0 || unexpectedTensorNames.Length > 0)
        {
            throw new InvalidDataException(
                $"GGUF tensor set does not match the repo-authored contract. Missing=[{string.Join(", ", missingTensorNames)}], unexpected=[{string.Join(", ", unexpectedTensorNames)}].");
        }

        var transformerProjectionWeights = new List<float[,]>(config.LayerCount * 7);
        var normScales = new List<float[]>(config.LayerCount * 2 + 1);
        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            normScales.Add(ReadVector(tensors[GetAttentionNormTensorName(layer)], config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetAttentionProjectionTensorName(layer, "q")], config.Dimension, config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetAttentionProjectionTensorName(layer, "k")], config.Dimension, config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetAttentionProjectionTensorName(layer, "v")], config.Dimension, config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetAttentionProjectionTensorName(layer, "out")], config.Dimension, config.Dimension));
            normScales.Add(ReadVector(tensors[GetFeedForwardNormTensorName(layer)], config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetFeedForwardProjectionTensorName(layer, "gate")], config.HiddenDimension, config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetFeedForwardProjectionTensorName(layer, "up")], config.HiddenDimension, config.Dimension));
            transformerProjectionWeights.Add(ReadMatrix(tensors[GetFeedForwardProjectionTensorName(layer, "down")], config.Dimension, config.HiddenDimension));
        }

        normScales.Add(ReadVector(tensors[OutputNormTensorName], config.Dimension));

        var snapshot = new BitNetPaperModelSnapshot(
            GetRequiredString(document.Metadata, "bitnetsharp.model_id"),
            GetRequiredInt32(document.Metadata, "bitnetsharp.bootstrap_seed"),
            config,
            vocabulary,
            GetRequiredInt32(document.Metadata, "bitnetsharp.max_response_tokens"),
            GetRequiredString(document.Metadata, "bitnetsharp.primary_language"),
            GetRequiredBool(document.Metadata, "bitnetsharp.enable_chain_buckets"),
            GetRequiredBool(document.Metadata, "bitnetsharp.enable_sequence_compression"),
            ReadAcceptanceThreshold(document.Metadata),
            ReadMatrix(tensors[TokenEmbeddingsTensorName], config.VocabSize, config.Dimension),
            transformerProjectionWeights,
            normScales,
            ReadMatrix(tensors[OutputTensorName], config.VocabSize, config.Dimension),
            DeserializeMemorizedResponses(GetRequiredString(document.Metadata, MemorizedResponsesMetadataKey)));
        var model = snapshot.Restore(verbosity);

        var bucketSidecarPath = GetBucketSidecarPath(path);
        if ((model.Options.EnableChainBuckets || model.Options.EnableSequenceCompression) && File.Exists(bucketSidecarPath))
        {
            model.LoadBucketTable(ChainBucketTableBinarySerializer.Load(bucketSidecarPath));
        }

        return model;
    }

    private static Dictionary<string, object> CreateMetadata(BitNetPaperModelSnapshot snapshot)
    {
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = ArchitectureName,
            ["general.name"] = snapshot.ModelId,
            ["general.alignment"] = (uint)32,
            ["bitnetsharp.format"] = FormatName,
            ["bitnetsharp.model_id"] = snapshot.ModelId,
            ["bitnetsharp.bootstrap_seed"] = snapshot.BootstrapSeed,
            [VocabularyMetadataKey] = JsonSerializer.Serialize(snapshot.Vocabulary),
            [MemorizedResponsesMetadataKey] = JsonSerializer.Serialize(snapshot.MemorizedResponses),
            ["bitnetsharp.max_response_tokens"] = snapshot.MaxResponseTokens,
            ["bitnetsharp.primary_language"] = snapshot.PrimaryLanguage,
            ["bitnetsharp.enable_chain_buckets"] = snapshot.EnableChainBuckets,
            ["bitnetsharp.enable_sequence_compression"] = snapshot.EnableSequenceCompression,
            ["bitnetsharp.chain_bucket_acceptance_threshold"] = snapshot.ChainBucketAcceptanceThreshold,
            ["bitnetsharp.config.vocab_size"] = snapshot.Config.VocabSize,
            ["bitnetsharp.config.dimension"] = snapshot.Config.Dimension,
            ["bitnetsharp.config.hidden_dimension"] = snapshot.Config.HiddenDimension,
            ["bitnetsharp.config.layer_count"] = snapshot.Config.LayerCount,
            ["bitnetsharp.config.head_count"] = snapshot.Config.HeadCount,
            ["bitnetsharp.config.max_sequence_length"] = snapshot.Config.MaxSequenceLength,
            ["bitnetsharp.config.rms_norm_epsilon"] = (double)snapshot.Config.RmsNormEpsilon
        };
    }

    private static IReadOnlyList<GgufTensor> CreateTensors(BitNetPaperModelSnapshot snapshot)
    {
        var tensors = new List<GgufTensor>
        {
            CreateMatrixTensor(TokenEmbeddingsTensorName, snapshot.TokenEmbeddings)
        };

        var projectionIndex = 0;
        var normIndex = 0;
        for (var layer = 0; layer < snapshot.Config.LayerCount; layer++)
        {
            tensors.Add(CreateVectorTensor(GetAttentionNormTensorName(layer), snapshot.NormScales[normIndex++]));
            tensors.Add(CreateMatrixTensor(GetAttentionProjectionTensorName(layer, "q"), snapshot.TransformerProjectionWeights[projectionIndex++]));
            tensors.Add(CreateMatrixTensor(GetAttentionProjectionTensorName(layer, "k"), snapshot.TransformerProjectionWeights[projectionIndex++]));
            tensors.Add(CreateMatrixTensor(GetAttentionProjectionTensorName(layer, "v"), snapshot.TransformerProjectionWeights[projectionIndex++]));
            tensors.Add(CreateMatrixTensor(GetAttentionProjectionTensorName(layer, "out"), snapshot.TransformerProjectionWeights[projectionIndex++]));
            tensors.Add(CreateVectorTensor(GetFeedForwardNormTensorName(layer), snapshot.NormScales[normIndex++]));
            tensors.Add(CreateMatrixTensor(GetFeedForwardProjectionTensorName(layer, "gate"), snapshot.TransformerProjectionWeights[projectionIndex++]));
            tensors.Add(CreateMatrixTensor(GetFeedForwardProjectionTensorName(layer, "up"), snapshot.TransformerProjectionWeights[projectionIndex++]));
            tensors.Add(CreateMatrixTensor(GetFeedForwardProjectionTensorName(layer, "down"), snapshot.TransformerProjectionWeights[projectionIndex++]));
        }

        tensors.Add(CreateVectorTensor(OutputNormTensorName, snapshot.NormScales[normIndex]));
        tensors.Add(CreateMatrixTensor(OutputTensorName, snapshot.OutputHeadWeights));
        return tensors;
    }

    private static void ValidateSnapshot(BitNetPaperModelSnapshot snapshot)
    {
        var expectedProjectionCount = snapshot.Config.LayerCount * 7;
        if (snapshot.TransformerProjectionWeights.Count != expectedProjectionCount)
        {
            throw new InvalidDataException(
                $"Expected {expectedProjectionCount} transformer projection tensors, but found {snapshot.TransformerProjectionWeights.Count}.");
        }

        var expectedNormCount = snapshot.Config.LayerCount * 2 + 1;
        if (snapshot.NormScales.Count != expectedNormCount)
        {
            throw new InvalidDataException($"Expected {expectedNormCount} norm tensors, but found {snapshot.NormScales.Count}.");
        }

        ValidateMatrixShape(snapshot.TokenEmbeddings, snapshot.Config.VocabSize, snapshot.Config.Dimension, TokenEmbeddingsTensorName);
        ValidateMatrixShape(snapshot.OutputHeadWeights, snapshot.Config.VocabSize, snapshot.Config.Dimension, OutputTensorName);
        foreach (var normScale in snapshot.NormScales)
        {
            if (normScale.Length != snapshot.Config.Dimension)
            {
                throw new InvalidDataException(
                    $"Expected norm scale length {snapshot.Config.Dimension}, but found {normScale.Length}.");
            }
        }
    }

    private static void ValidateMetadata(IReadOnlyDictionary<string, object> metadata)
    {
        if (!string.Equals(GetRequiredString(metadata, "bitnetsharp.format"), FormatName, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Unsupported BitNet GGUF format '{GetRequiredString(metadata, "bitnetsharp.format")}'.");
        }

        if (!string.Equals(GetRequiredString(metadata, "general.architecture"), ArchitectureName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Unsupported GGUF architecture '{GetRequiredString(metadata, "general.architecture")}'.");
        }
    }

    private static BitNetConfig ReadConfig(IReadOnlyDictionary<string, object> metadata)
    {
        return new BitNetConfig(
            vocabSize: GetRequiredInt32(metadata, "bitnetsharp.config.vocab_size"),
            dimension: GetRequiredInt32(metadata, "bitnetsharp.config.dimension"),
            hiddenDimension: GetRequiredInt32(metadata, "bitnetsharp.config.hidden_dimension"),
            layerCount: GetRequiredInt32(metadata, "bitnetsharp.config.layer_count"),
            headCount: GetRequiredInt32(metadata, "bitnetsharp.config.head_count"),
            maxSequenceLength: GetRequiredInt32(metadata, "bitnetsharp.config.max_sequence_length"),
            rmsNormEpsilon: (float)GetRequiredDouble(metadata, "bitnetsharp.config.rms_norm_epsilon"));
    }

    private static string[] DeserializeVocabulary(string json) =>
        JsonSerializer.Deserialize<string[]>(json)
        ?? throw new InvalidDataException("Could not deserialize the GGUF vocabulary payload.");

    private static Dictionary<string, int[]> DeserializeMemorizedResponses(string json)
    {
        var result = JsonSerializer.Deserialize<Dictionary<string, int[]>>(json)
            ?? throw new InvalidDataException("Could not deserialize the GGUF memorized-response payload.");
        return new Dictionary<string, int[]>(result, StringComparer.Ordinal);
    }

    private static double ReadAcceptanceThreshold(IReadOnlyDictionary<string, object> metadata)
    {
        var threshold = GetRequiredDouble(metadata, "bitnetsharp.chain_bucket_acceptance_threshold");
        return threshold > 0d ? threshold : 0.85d;
    }

    private static IReadOnlyList<string> CreateExpectedTensorNames(BitNetConfig config)
    {
        var names = new List<string> { TokenEmbeddingsTensorName };
        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            names.Add(GetAttentionNormTensorName(layer));
            names.Add(GetAttentionProjectionTensorName(layer, "q"));
            names.Add(GetAttentionProjectionTensorName(layer, "k"));
            names.Add(GetAttentionProjectionTensorName(layer, "v"));
            names.Add(GetAttentionProjectionTensorName(layer, "out"));
            names.Add(GetFeedForwardNormTensorName(layer));
            names.Add(GetFeedForwardProjectionTensorName(layer, "gate"));
            names.Add(GetFeedForwardProjectionTensorName(layer, "up"));
            names.Add(GetFeedForwardProjectionTensorName(layer, "down"));
        }

        names.Add(OutputNormTensorName);
        names.Add(OutputTensorName);
        return names;
    }

    private static GgufTensor CreateMatrixTensor(string name, float[,] matrix)
    {
        return new GgufTensor(name, [matrix.GetLength(0), matrix.GetLength(1)], FlattenMatrix(matrix));
    }

    private static GgufTensor CreateVectorTensor(string name, IReadOnlyList<float> vector)
    {
        return new GgufTensor(name, [vector.Count], BitNetPaperModelSnapshot.CloneVector(vector));
    }

    private static float[,] ReadMatrix(GgufTensor tensor, int expectedRows, int expectedColumns)
    {
        if (tensor.Dimensions.Count != 2)
        {
            throw new InvalidDataException($"GGUF tensor '{tensor.Name}' must be rank 2.");
        }

        if (tensor.Dimensions[0] != expectedRows || tensor.Dimensions[1] != expectedColumns)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{tensor.Name}' expected shape [{expectedRows}, {expectedColumns}] but found [{tensor.Dimensions[0]}, {tensor.Dimensions[1]}].");
        }

        var matrix = new float[expectedRows, expectedColumns];
        var offset = 0;
        for (var row = 0; row < expectedRows; row++)
        {
            for (var column = 0; column < expectedColumns; column++)
            {
                matrix[row, column] = tensor.Data[offset++];
            }
        }

        return matrix;
    }

    private static float[] ReadVector(GgufTensor tensor, int expectedLength)
    {
        if (tensor.Dimensions.Count != 1)
        {
            throw new InvalidDataException($"GGUF tensor '{tensor.Name}' must be rank 1.");
        }

        if (tensor.Dimensions[0] != expectedLength)
        {
            throw new InvalidDataException(
                $"GGUF tensor '{tensor.Name}' expected length {expectedLength} but found {tensor.Dimensions[0]}.");
        }

        return [.. tensor.Data];
    }

    private static float[] FlattenMatrix(float[,] matrix)
    {
        var data = new float[matrix.Length];
        var offset = 0;
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var column = 0; column < matrix.GetLength(1); column++)
            {
                data[offset++] = matrix[row, column];
            }
        }

        return data;
    }

    private static string GetAttentionNormTensorName(int layer) => $"blk.{layer}.attn_norm.weight";

    private static string GetFeedForwardNormTensorName(int layer) => $"blk.{layer}.ffn_norm.weight";

    private static string GetAttentionProjectionTensorName(int layer, string suffix) => $"blk.{layer}.attn_{suffix}.weight";

    private static string GetFeedForwardProjectionTensorName(int layer, string suffix) => $"blk.{layer}.ffn_{suffix}.weight";

    private static string GetBucketSidecarPath(string ggufPath)
    {
        var directory = Path.GetDirectoryName(ggufPath);
        var baseName = Path.GetFileNameWithoutExtension(ggufPath);
        var fileName = $"{baseName}.chain-buckets.bin";
        return string.IsNullOrWhiteSpace(directory)
            ? fileName
            : Path.Combine(directory, fileName);
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

    private static void ValidateMatrixShape(float[,] matrix, int expectedRows, int expectedColumns, string name)
    {
        if (matrix.GetLength(0) != expectedRows || matrix.GetLength(1) != expectedColumns)
        {
            throw new InvalidDataException(
                $"Tensor '{name}' expected shape [{expectedRows}, {expectedColumns}] but found [{matrix.GetLength(0)}, {matrix.GetLength(1)}].");
        }
    }

    private static string GetRequiredString(IReadOnlyDictionary<string, object> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var value) || value is not string text || string.IsNullOrWhiteSpace(text))
        {
            throw new InvalidDataException($"Missing required GGUF string metadata key '{key}'.");
        }

        return text;
    }

    private static bool GetRequiredBool(IReadOnlyDictionary<string, object> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var value) || value is not bool boolean)
        {
            throw new InvalidDataException($"Missing required GGUF boolean metadata key '{key}'.");
        }

        return boolean;
    }

    private static int GetRequiredInt32(IReadOnlyDictionary<string, object> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var value))
        {
            throw new InvalidDataException($"Missing required GGUF integer metadata key '{key}'.");
        }

        return value switch
        {
            int signedInt32 => signedInt32,
            uint unsignedInt32 when unsignedInt32 <= int.MaxValue => (int)unsignedInt32,
            long signedInt64 when signedInt64 >= int.MinValue && signedInt64 <= int.MaxValue => (int)signedInt64,
            ulong unsignedInt64 when unsignedInt64 <= int.MaxValue => (int)unsignedInt64,
            _ => throw new InvalidDataException($"GGUF metadata key '{key}' is not a supported integer value.")
        };
    }

    private static double GetRequiredDouble(IReadOnlyDictionary<string, object> metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var value))
        {
            throw new InvalidDataException($"Missing required GGUF floating-point metadata key '{key}'.");
        }

        return value switch
        {
            double float64 => float64,
            float float32 => float32,
            _ => throw new InvalidDataException($"GGUF metadata key '{key}' is not a supported floating-point value.")
        };
    }
}
