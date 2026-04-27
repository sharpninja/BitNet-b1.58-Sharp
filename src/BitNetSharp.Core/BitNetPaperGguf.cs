using System.Runtime.InteropServices;
using System.Text.Json;
using BitNetSharp.Core.Bucketing;
using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Serialization.Gguf;
using Microsoft.Extensions.Logging;

namespace BitNetSharp.Core;

public static class BitNetPaperGguf
{
    private const string FormatNameV1 = "bitnet-b1.58-sharp.gguf.v1";
    private const string FormatNameV2 = "bitnet-b1.58-sharp.gguf.v2";
    private const uint FormatVersionV1 = 1;
    private const uint FormatVersionV2 = 2;
    private const string FormatVersionMetadataKey = "bitnetsharp.format_version";
    private const string ArchitectureName = "bitnetsharp";
    private const string TokenEmbeddingsTensorName = "token_embeddings";
    private const string OutputNormTensorName = "output_norm.weight";
    private const string OutputTensorName = "output.weight";
    private const string VocabularyMetadataKey = "bitnetsharp.vocabulary";
    private const string MemorizedResponsesMetadataKey = "bitnetsharp.memorized_responses";
    private const int DefaultBootstrapSeed = 42;

    public static void Save(BitNetPaperModel model, string path)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var metadata = CreateMetadataFromModel(model);
        var tensors = CreateStreamingTensors(model);
        GgufStreamingWriter.Write(path, metadata, tensors);
        SaveBucketSidecar(model.BucketTable, GetBucketSidecarPath(path));
        SaveHeatMapSidecar(model.RecallHeatMap, GetHeatMapSidecarPath(path));
    }

    public static BitNetPaperModel Load(
        string path,
        ILogger<BitNetPaperModel> logger,
        ILoggerFactory loggerFactory,
        VerbosityLevel verbosity = VerbosityLevel.Normal)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(logger);
        ArgumentNullException.ThrowIfNull(loggerFactory);

        using var reader = GgufStreamingReader.Open(path);
        ValidateMetadata(reader.Metadata);

        var config = ReadConfig(reader.Metadata);
        // KV-FU5: serve loads default to Int8 KV after the Section B
        // measurements showed strict win across every Bonsai workload
        // (short-ctx 2.6x total + 3.1x TTFT + 3% decode; long-ctx 1.45x
        // total + 1.63x TTFT + 9% decode). BitNetConfig() default stays Fp32
        // for backwards compat with direct callers. Env var
        // BITNETSHARP_KV_CACHE_QUANTIZATION can still force either side
        // (e.g. set to Fp32 to opt out of the new default).
        var kvOverride = BitNetOptions.KvCacheQuantizationEnvOverride;
        var kvResolved = kvOverride ?? KvCacheQuantization.Int8;
        if (kvResolved != config.KvCacheQuantization)
        {
            config = new BitNetConfig(
                vocabSize: config.VocabSize,
                dimension: config.Dimension,
                hiddenDimension: config.HiddenDimension,
                layerCount: config.LayerCount,
                headCount: config.HeadCount,
                maxSequenceLength: config.MaxSequenceLength,
                rmsNormEpsilon: config.RmsNormEpsilon,
                kvHeadCount: config.KvHeadCount,
                ropeTheta: config.RopeTheta,
                kvCacheQuantization: kvResolved);
        }
        logger.LogInformation(
            kvOverride.HasValue
                ? "KvCacheQuantization={Value} (override applied via {Var})"
                : "KvCacheQuantization={Value} (serve default; set {Var}=Fp32 to opt out)",
            kvResolved,
            BitNetOptions.KvCacheQuantizationEnvVar);
        var vocabulary = DeserializeVocabulary(GetRequiredString(reader.Metadata, VocabularyMetadataKey));
        var memorizedResponses = DeserializeMemorizedResponses(GetRequiredString(reader.Metadata, MemorizedResponsesMetadataKey));

        var tensorByName = reader.Tensors.ToDictionary(static t => t.Name, StringComparer.Ordinal);
        var expectedTensorNames = CreateExpectedTensorNames(config);
        var missingTensorNames = expectedTensorNames.Where(name => !tensorByName.ContainsKey(name)).ToArray();
        var unexpectedTensorNames = tensorByName.Keys.Where(name => !expectedTensorNames.Contains(name, StringComparer.Ordinal)).ToArray();
        if (missingTensorNames.Length > 0 || unexpectedTensorNames.Length > 0)
        {
            throw new InvalidDataException(
                $"GGUF tensor set does not match the repo-authored contract. Missing=[{string.Join(", ", missingTensorNames)}], unexpected=[{string.Join(", ", unexpectedTensorNames)}].");
        }

        var options = new BitNetOptions(
            [.. vocabulary],
            verbosity,
            GetRequiredInt32(reader.Metadata, "bitnetsharp.max_response_tokens"),
            GetRequiredString(reader.Metadata, "bitnetsharp.primary_language"),
            GetRequiredBool(reader.Metadata, "bitnetsharp.enable_chain_buckets"),
            GetRequiredBool(reader.Metadata, "bitnetsharp.enable_sequence_compression"),
            ReadAcceptanceThreshold(reader.Metadata),
            ReadOptionalBool(reader.Metadata, "bitnetsharp.enable_recall_heat_map", defaultValue: true),
            UseIntegerForward: BitNetOptions.IntegerForwardEnvDefault);

        var bootstrapSeed = GetRequiredInt32(reader.Metadata, "bitnetsharp.bootstrap_seed");
        var model = new BitNetPaperModel(options, logger, loggerFactory, config, bootstrapSeed);

        var formatVersion = DetectFormatVersion(reader.Metadata);

        // Stream tensors one at a time. v2 (packed): each BitLinear read is
        // O(packed_stride * out_dim) ~= 1/20th FP32; Gamma round-trips
        // bit-exact (no quantize pass). v1 (FP32): each ReadMatrix allocates
        // the full FP32 buffer, QuantizeFromFullPrecision immediately repacks
        // into trits, and the float[,] becomes GC-eligible before the next
        // tensor is read. In either case peak additional RAM is bounded by
        // the largest single projection.
        model.ImportTokenEmbeddings(reader.ReadMatrix(tensorByName[TokenEmbeddingsTensorName], config.VocabSize, config.Dimension));

        var linearLayers = model.GetTransformerBitLinearLayers().ToArray();
        var normLayers = model.GetNormLayers().ToArray();
        var kvDim = config.KvHeadCount * config.HeadDimension;
        var linearIndex = 0;
        var normIndex = 0;
        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            normLayers[normIndex++].ImportScale(reader.ReadVector(tensorByName[GetAttentionNormTensorName(layer)], config.Dimension));
            ImportBitLinear(reader, tensorByName[GetAttentionProjectionTensorName(layer, "q")], linearLayers[linearIndex++], config.Dimension, config.Dimension, formatVersion);
            ImportBitLinear(reader, tensorByName[GetAttentionProjectionTensorName(layer, "k")], linearLayers[linearIndex++], kvDim, config.Dimension, formatVersion);
            ImportBitLinear(reader, tensorByName[GetAttentionProjectionTensorName(layer, "v")], linearLayers[linearIndex++], kvDim, config.Dimension, formatVersion);
            ImportBitLinear(reader, tensorByName[GetAttentionProjectionTensorName(layer, "out")], linearLayers[linearIndex++], config.Dimension, config.Dimension, formatVersion);
            normLayers[normIndex++].ImportScale(reader.ReadVector(tensorByName[GetFeedForwardNormTensorName(layer)], config.Dimension));
            ImportBitLinear(reader, tensorByName[GetFeedForwardProjectionTensorName(layer, "gate")], linearLayers[linearIndex++], config.HiddenDimension, config.Dimension, formatVersion);
            ImportBitLinear(reader, tensorByName[GetFeedForwardProjectionTensorName(layer, "up")], linearLayers[linearIndex++], config.HiddenDimension, config.Dimension, formatVersion);
            ImportBitLinear(reader, tensorByName[GetFeedForwardProjectionTensorName(layer, "down")], linearLayers[linearIndex++], config.Dimension, config.HiddenDimension, formatVersion);
        }

        normLayers[normIndex].ImportScale(reader.ReadVector(tensorByName[OutputNormTensorName], config.Dimension));
        ImportBitLinear(reader, tensorByName[OutputTensorName], model.GetOutputHead(), config.VocabSize, config.Dimension, formatVersion);

        model.ImportMemorizedResponses(memorizedResponses);

        var bucketSidecarPath = GetBucketSidecarPath(path);
        if ((model.Options.EnableChainBuckets || model.Options.EnableSequenceCompression) && File.Exists(bucketSidecarPath))
        {
            model.LoadBucketTable(ChainBucketTableBinarySerializer.Load(bucketSidecarPath));
        }

        var heatMapSidecarPath = GetHeatMapSidecarPath(path);
        if (model.RecallHeatMap is not null && File.Exists(heatMapSidecarPath))
        {
            model.RecallHeatMap.MergeFrom(BucketRecallHeatMapSerializer.Load(heatMapSidecarPath));
        }

        return model;
    }

    private static Dictionary<string, object> CreateMetadataFromModel(BitNetPaperModel model)
    {
        var config = model.Config;
        var options = model.Options;
        var memorized = BitNetPaperModelSnapshot.CloneMemorizedResponses(model.ExportMemorizedResponses());
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = ArchitectureName,
            ["general.name"] = model.ModelId,
            ["general.alignment"] = (uint)32,
            ["bitnetsharp.format"] = FormatNameV2,
            [FormatVersionMetadataKey] = FormatVersionV2,
            ["bitnetsharp.model_id"] = model.ModelId,
            ["bitnetsharp.bootstrap_seed"] = DefaultBootstrapSeed,
            [VocabularyMetadataKey] = JsonSerializer.Serialize(options.Vocabulary),
            [MemorizedResponsesMetadataKey] = JsonSerializer.Serialize(memorized),
            ["bitnetsharp.max_response_tokens"] = options.MaxResponseTokens,
            ["bitnetsharp.primary_language"] = options.PrimaryLanguage,
            ["bitnetsharp.enable_chain_buckets"] = options.EnableChainBuckets,
            ["bitnetsharp.enable_sequence_compression"] = options.EnableSequenceCompression,
            ["bitnetsharp.enable_recall_heat_map"] = options.EnableRecallHeatMap,
            ["bitnetsharp.chain_bucket_acceptance_threshold"] = options.ChainBucketAcceptanceThreshold,
            ["bitnetsharp.config.vocab_size"] = config.VocabSize,
            ["bitnetsharp.config.dimension"] = config.Dimension,
            ["bitnetsharp.config.hidden_dimension"] = config.HiddenDimension,
            ["bitnetsharp.config.layer_count"] = config.LayerCount,
            ["bitnetsharp.config.head_count"] = config.HeadCount,
            ["bitnetsharp.config.kv_head_count"] = config.KvHeadCount,
            ["bitnetsharp.config.rope_theta"] = (double)config.RopeTheta,
            ["bitnetsharp.config.max_sequence_length"] = config.MaxSequenceLength,
            ["bitnetsharp.config.rms_norm_epsilon"] = (double)config.RmsNormEpsilon
        };
    }

    private static IReadOnlyList<GgufStreamingTensor> CreateStreamingTensors(BitNetPaperModel model)
    {
        var config = model.Config;
        var tensors = new List<GgufStreamingTensor>(config.LayerCount * 9 + 3);

        var tokenEmbeddings = model.GetTokenEmbeddingsMatrix();
        tensors.Add(CreateStreamingMatrixTensor(TokenEmbeddingsTensorName, tokenEmbeddings));

        var linearLayers = model.GetTransformerBitLinearLayers().ToArray();
        var normLayers = model.GetNormLayers().ToArray();

        var expectedLinearCount = config.LayerCount * 7;
        if (linearLayers.Length != expectedLinearCount)
        {
            throw new InvalidDataException(
                $"Expected {expectedLinearCount} transformer BitLinear layers, but found {linearLayers.Length}.");
        }

        var expectedNormCount = config.LayerCount * 2 + 1;
        if (normLayers.Length != expectedNormCount)
        {
            throw new InvalidDataException($"Expected {expectedNormCount} norm layers, but found {normLayers.Length}.");
        }

        var linearIndex = 0;
        var normIndex = 0;
        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            tensors.Add(CreateStreamingVectorTensor(GetAttentionNormTensorName(layer), normLayers[normIndex++].ExportScale()));
            tensors.Add(CreateStreamingBitLinearTensor(GetAttentionProjectionTensorName(layer, "q"), linearLayers[linearIndex++]));
            tensors.Add(CreateStreamingBitLinearTensor(GetAttentionProjectionTensorName(layer, "k"), linearLayers[linearIndex++]));
            tensors.Add(CreateStreamingBitLinearTensor(GetAttentionProjectionTensorName(layer, "v"), linearLayers[linearIndex++]));
            tensors.Add(CreateStreamingBitLinearTensor(GetAttentionProjectionTensorName(layer, "out"), linearLayers[linearIndex++]));
            tensors.Add(CreateStreamingVectorTensor(GetFeedForwardNormTensorName(layer), normLayers[normIndex++].ExportScale()));
            tensors.Add(CreateStreamingBitLinearTensor(GetFeedForwardProjectionTensorName(layer, "gate"), linearLayers[linearIndex++]));
            tensors.Add(CreateStreamingBitLinearTensor(GetFeedForwardProjectionTensorName(layer, "up"), linearLayers[linearIndex++]));
            tensors.Add(CreateStreamingBitLinearTensor(GetFeedForwardProjectionTensorName(layer, "down"), linearLayers[linearIndex++]));
        }

        tensors.Add(CreateStreamingVectorTensor(OutputNormTensorName, normLayers[normIndex].ExportScale()));
        tensors.Add(CreateStreamingBitLinearTensor(OutputTensorName, model.GetOutputHead()));
        return tensors;
    }

    private static GgufStreamingTensor CreateStreamingMatrixTensor(string name, float[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var payloadBytes = checked((long)rows * cols * sizeof(float));
        var captured = matrix;
        return new GgufStreamingTensor(
            name,
            new[] { rows, cols },
            GgufTensorTypes.Float32,
            payloadBytes,
            writer =>
            {
                var rowBuffer = new float[cols];
                for (var row = 0; row < rows; row++)
                {
                    for (var column = 0; column < cols; column++)
                    {
                        rowBuffer[column] = captured[row, column];
                    }

                    writer.Write(MemoryMarshal.AsBytes(rowBuffer.AsSpan()));
                }
            });
    }

    private static GgufStreamingTensor CreateStreamingVectorTensor(string name, float[] values)
    {
        var captured = values;
        var payloadBytes = checked((long)values.Length * sizeof(float));
        return new GgufStreamingTensor(
            name,
            new[] { values.Length },
            GgufTensorTypes.Float32,
            payloadBytes,
            writer => writer.Write(MemoryMarshal.AsBytes(captured.AsSpan())));
    }

    private static GgufStreamingTensor CreateStreamingBitLinearTensor(string name, Layers.BitLinear layer)
    {
        var rows = layer.Config.OutputDimension;
        var cols = layer.Config.InputDimension;
        // v2 layout: [float32 gamma][byte[] packed_trits]. 5 trits per byte.
        var packedBytes = checked((long)layer.PackedStride * rows);
        var payloadBytes = checked(sizeof(float) + packedBytes);
        var captured = layer;
        return new GgufStreamingTensor(
            name,
            new[] { rows, cols },
            GgufTensorTypes.BitNetSharpPackedTernary,
            payloadBytes,
            writer =>
            {
                writer.Write(captured.Gamma);
                captured.WritePackedTritsTo(writer);
            });
    }

    private static Dictionary<string, object> CreateMetadata(BitNetPaperModelSnapshot snapshot)
    {
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = ArchitectureName,
            ["general.name"] = snapshot.ModelId,
            ["general.alignment"] = (uint)32,
            ["bitnetsharp.format"] = FormatNameV2,
            [FormatVersionMetadataKey] = FormatVersionV2,
            ["bitnetsharp.model_id"] = snapshot.ModelId,
            ["bitnetsharp.bootstrap_seed"] = snapshot.BootstrapSeed,
            [VocabularyMetadataKey] = JsonSerializer.Serialize(snapshot.Vocabulary),
            [MemorizedResponsesMetadataKey] = JsonSerializer.Serialize(snapshot.MemorizedResponses),
            ["bitnetsharp.max_response_tokens"] = snapshot.MaxResponseTokens,
            ["bitnetsharp.primary_language"] = snapshot.PrimaryLanguage,
            ["bitnetsharp.enable_chain_buckets"] = snapshot.EnableChainBuckets,
            ["bitnetsharp.enable_sequence_compression"] = snapshot.EnableSequenceCompression,
            ["bitnetsharp.enable_recall_heat_map"] = snapshot.EnableRecallHeatMap,
            ["bitnetsharp.chain_bucket_acceptance_threshold"] = snapshot.ChainBucketAcceptanceThreshold,
            ["bitnetsharp.config.vocab_size"] = snapshot.Config.VocabSize,
            ["bitnetsharp.config.dimension"] = snapshot.Config.Dimension,
            ["bitnetsharp.config.hidden_dimension"] = snapshot.Config.HiddenDimension,
            ["bitnetsharp.config.layer_count"] = snapshot.Config.LayerCount,
            ["bitnetsharp.config.head_count"] = snapshot.Config.HeadCount,
            ["bitnetsharp.config.kv_head_count"] = snapshot.Config.KvHeadCount,
            ["bitnetsharp.config.rope_theta"] = (double)snapshot.Config.RopeTheta,
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
        var format = GetRequiredString(metadata, "bitnetsharp.format");
        if (!string.Equals(format, FormatNameV1, StringComparison.Ordinal)
            && !string.Equals(format, FormatNameV2, StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Unsupported BitNet GGUF format '{format}'.");
        }

        if (!string.Equals(GetRequiredString(metadata, "general.architecture"), ArchitectureName, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Unsupported GGUF architecture '{GetRequiredString(metadata, "general.architecture")}'.");
        }
    }

    private static uint DetectFormatVersion(IReadOnlyDictionary<string, object> metadata)
    {
        if (metadata.TryGetValue(FormatVersionMetadataKey, out var raw))
        {
            return raw switch
            {
                uint u => u,
                int i when i >= 0 => (uint)i,
                ulong ul when ul <= uint.MaxValue => (uint)ul,
                long l when l >= 0 && l <= uint.MaxValue => (uint)l,
                _ => throw new InvalidDataException($"GGUF metadata key '{FormatVersionMetadataKey}' is not a supported integer value."),
            };
        }

        // Legacy files predating the version key are v1 (FP32-on-disk).
        var format = GetRequiredString(metadata, "bitnetsharp.format");
        return string.Equals(format, FormatNameV2, StringComparison.Ordinal)
            ? FormatVersionV2
            : FormatVersionV1;
    }

    private static void ImportBitLinear(
        GgufStreamingReader reader,
        GgufTensorInfo tensorInfo,
        Layers.BitLinear layer,
        int outputDim,
        int inputDim,
        uint formatVersion)
    {
        if (formatVersion >= FormatVersionV2 && tensorInfo.TensorType == GgufTensorTypes.BitNetSharpPackedTernary)
        {
            var (packed, gamma) = reader.ReadPackedTernary(tensorInfo, outputDim, inputDim);
            layer.ImportPacked(packed, gamma);
            return;
        }

        // v1 FP32 path. Also covers v2 files that emitted F32 tensors for
        // back-compat (none do today, but the dispatch is cheap).
        layer.QuantizeFromFullPrecision(reader.ReadMatrix(tensorInfo, outputDim, inputDim));
    }

    private static BitNetConfig ReadConfig(IReadOnlyDictionary<string, object> metadata)
    {
        int headCount = GetRequiredInt32(metadata, "bitnetsharp.config.head_count");
        int kvHeadCount = ReadOptionalInt32(metadata, "bitnetsharp.config.kv_head_count", headCount);
        float ropeTheta = (float)ReadOptionalDouble(metadata, "bitnetsharp.config.rope_theta", 10_000d);
        return new BitNetConfig(
            vocabSize: GetRequiredInt32(metadata, "bitnetsharp.config.vocab_size"),
            dimension: GetRequiredInt32(metadata, "bitnetsharp.config.dimension"),
            hiddenDimension: GetRequiredInt32(metadata, "bitnetsharp.config.hidden_dimension"),
            layerCount: GetRequiredInt32(metadata, "bitnetsharp.config.layer_count"),
            headCount: headCount,
            maxSequenceLength: GetRequiredInt32(metadata, "bitnetsharp.config.max_sequence_length"),
            rmsNormEpsilon: (float)GetRequiredDouble(metadata, "bitnetsharp.config.rms_norm_epsilon"),
            kvHeadCount: kvHeadCount,
            ropeTheta: ropeTheta);
    }

    private static int ReadOptionalInt32(IReadOnlyDictionary<string, object> metadata, string key, int defaultValue)
    {
        if (!metadata.TryGetValue(key, out var raw)) return defaultValue;
        return raw switch
        {
            int i => i,
            uint u => checked((int)u),
            long l => checked((int)l),
            ulong ul => checked((int)ul),
            _ => defaultValue,
        };
    }

    private static double ReadOptionalDouble(IReadOnlyDictionary<string, object> metadata, string key, double defaultValue)
    {
        if (!metadata.TryGetValue(key, out var raw)) return defaultValue;
        return raw switch
        {
            double d => d,
            float f => f,
            _ => defaultValue,
        };
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

    private static string GetHeatMapSidecarPath(string ggufPath)
    {
        var directory = Path.GetDirectoryName(ggufPath);
        var baseName = Path.GetFileNameWithoutExtension(ggufPath);
        var fileName = $"{baseName}.recall-heatmap.bin";
        return string.IsNullOrWhiteSpace(directory)
            ? fileName
            : Path.Combine(directory, fileName);
    }

    private static void SaveHeatMapSidecar(BucketRecallHeatMap? heatMap, string path)
    {
        if (heatMap is null)
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }

            return;
        }

        BucketRecallHeatMapSerializer.Save(heatMap, path);
    }

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

    private static bool ReadOptionalBool(IReadOnlyDictionary<string, object> metadata, string key, bool defaultValue)
    {
        if (metadata.TryGetValue(key, out var value) && value is bool boolean)
        {
            return boolean;
        }

        return defaultValue;
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
