using System;
using System.Collections.Generic;
using System.IO;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Serialization.Gguf;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Core.Converters;

/// <summary>
/// Converts a PrismML Ternary-Bonsai-8B-style GGUF (Qwen3 architecture,
/// Prism Q2_0 body weights) into a BitNetSharp ternary model. The conversion
/// is lossy at the per-weight level: the quaternary Q2_0 +2d state collapses
/// to +1*Gamma (magnitude preserved at the tensor level via the analytic
/// Gamma formula). token_embd.weight and output.weight are discarded because
/// Bonsai's 151936-entry BPE vocab does not match BitNetSharp's word-level
/// vocabulary; the caller's fresh bootstrap-seeded embeddings stay in place.
/// </summary>
public static class Qwen3BonsaiConverter
{
    public const string ExpectedArchitecture = "qwen3";
    private const string ArchMetaKey = "general.architecture";
    private const int DefaultMaxSequenceLength = 65536;
    private const float DefaultRopeTheta = 1_000_000f;
    private const float DefaultRmsNormEpsilon = 1e-6f;

    public static BitNetPaperModel Convert(string sourcePath, BitNetOptions targetOptions, int seed = 42)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(targetOptions);
        var doc = GgufExtendedReader.Read(sourcePath);
        return Convert(doc, targetOptions, seed);
    }

    public static BitNetPaperModel Convert(GgufExtendedDocument source, BitNetOptions targetOptions, int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(targetOptions);

        VerifyArchitecture(source);

        // Build a probe model just to read the caller-effective vocab size.
        var probe = new BitNetPaperModel(targetOptions, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, config: null, seed: seed);
        var qwenConfig = DeriveConfigFromMetadata(source, probe.Config.VocabSize);
        var model = new BitNetPaperModel(targetOptions, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, qwenConfig, seed: seed);
        Import(source, model);
        return model;
    }

    private static BitNetConfig DeriveConfigFromMetadata(GgufExtendedDocument source, int vocabSize)
    {
        int blockCount = RequireUInt32(source, "qwen3.block_count");
        int embeddingLength = RequireUInt32(source, "qwen3.embedding_length");
        int headCount = RequireUInt32(source, "qwen3.attention.head_count");
        int kvHeadCount = RequireUInt32(source, "qwen3.attention.head_count_kv");
        int feedForwardLength = RequireUInt32(source, "qwen3.feed_forward_length");

        int maxSeqLen = TryGetUInt32(source, "qwen3.context_length") ?? DefaultMaxSequenceLength;
        float ropeTheta = TryGetFloat32(source, "qwen3.rope.freq_base") ?? DefaultRopeTheta;
        float epsilon = TryGetFloat32(source, "qwen3.attention.layer_norm_rms_epsilon") ?? DefaultRmsNormEpsilon;

        return new BitNetConfig(
            vocabSize: vocabSize,
            dimension: embeddingLength,
            hiddenDimension: feedForwardLength,
            layerCount: blockCount,
            headCount: headCount,
            maxSequenceLength: maxSeqLen,
            rmsNormEpsilon: epsilon,
            kvHeadCount: kvHeadCount,
            ropeTheta: ropeTheta);
    }

    private static int RequireUInt32(GgufExtendedDocument source, string key)
    {
        int? value = TryGetUInt32(source, key);
        if (value is null)
        {
            throw new InvalidDataException(
                $"GGUF missing required metadata '{key}' for Qwen3 import.");
        }
        return value.Value;
    }

    private static int? TryGetUInt32(GgufExtendedDocument source, string key)
    {
        if (!source.Metadata.TryGetValue(key, out var raw)) return null;
        return raw switch
        {
            uint u => checked((int)u),
            int i => i,
            ulong ul => checked((int)ul),
            long l => checked((int)l),
            _ => null,
        };
    }

    private static float? TryGetFloat32(GgufExtendedDocument source, string key)
    {
        if (!source.Metadata.TryGetValue(key, out var raw)) return null;
        return raw switch
        {
            float f => f,
            double d => (float)d,
            _ => null,
        };
    }

    public static void Import(GgufExtendedDocument source, BitNetPaperModel target)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(target);

        VerifyArchitecture(source);

        var byName = new Dictionary<string, GgufRawTensor>(source.Tensors.Count, StringComparer.Ordinal);
        foreach (var t in source.Tensors)
        {
            byName[t.Name] = t;
        }

        var transformer = target.Transformer;
        var config = target.Config;

        // Verify blk.{layerCount-1}.* exist and blk.{layerCount}.* does not.
        string lastBlockKey = $"blk.{config.LayerCount - 1}.attn_q.weight";
        string overflowKey = $"blk.{config.LayerCount}.attn_q.weight";
        if (!byName.ContainsKey(lastBlockKey))
        {
            throw new InvalidDataException(
                $"GGUF missing tensor '{lastBlockKey}'. Target model has {config.LayerCount} layers; source has fewer.");
        }
        if (byName.ContainsKey(overflowKey))
        {
            throw new InvalidDataException(
                $"GGUF has more layers than target model ({config.LayerCount}). Found tensor '{overflowKey}'.");
        }

        // Final output norm
        ImportNormIntoRmsNorm(byName, "output_norm.weight", transformer.FinalNorm);

        for (int i = 0; i < config.LayerCount; i++)
        {
            string p = $"blk.{i}.";
            var layer = transformer.Layers[i];

            ImportNormIntoRmsNorm(byName, p + "attn_norm.weight", layer.PreAttentionNorm);
            ImportQ2_0IntoBitLinear(byName, p + "attn_q.weight", layer.Attention.QueryProjection);
            ImportQ2_0IntoBitLinear(byName, p + "attn_k.weight", layer.Attention.KeyProjection);
            ImportQ2_0IntoBitLinear(byName, p + "attn_v.weight", layer.Attention.ValueProjection);
            ImportQ2_0IntoBitLinear(byName, p + "attn_output.weight", layer.Attention.OutputProjection);

            ImportNormIntoRmsNorm(byName, p + "ffn_norm.weight", layer.PreFeedForwardNorm);
            ImportQ2_0IntoBitLinear(byName, p + "ffn_gate.weight", layer.FeedForward.GateProjection);
            ImportQ2_0IntoBitLinear(byName, p + "ffn_up.weight", layer.FeedForward.UpProjection);
            ImportQ2_0IntoBitLinear(byName, p + "ffn_down.weight", layer.FeedForward.DownProjection);
        }

        // token_embd.weight and output.weight are intentionally discarded.
        // BitNetSharp's word-level vocabulary would need a separate alignment
        // pass (e.g. fine-tune via TrainingCommand) to project Bonsai's BPE
        // space onto ours.
    }

    private static void VerifyArchitecture(GgufExtendedDocument source)
    {
        if (!source.Metadata.TryGetValue(ArchMetaKey, out var archObj) || archObj is not string arch)
        {
            throw new InvalidDataException($"GGUF missing required '{ArchMetaKey}' metadata.");
        }
        if (!string.Equals(arch, ExpectedArchitecture, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Expected architecture '{ExpectedArchitecture}', got '{arch}'. Only Qwen3-shaped GGUFs are supported.");
        }
    }

    private static void ImportNormIntoRmsNorm(IReadOnlyDictionary<string, GgufRawTensor> byName, string key, RmsNorm norm)
    {
        if (!byName.TryGetValue(key, out var tensor))
        {
            throw new InvalidDataException($"GGUF missing tensor '{key}'.");
        }

        int dim = norm.Dimension;
        if (tensor.Dimensions.Count != 1 || tensor.Dimensions[0] != dim)
        {
            throw new InvalidDataException(
                $"Tensor '{key}' shape {FormatDims(tensor.Dimensions)} does not match expected [{dim}].");
        }

        var scale = new float[dim];
        switch (tensor.GgmlType)
        {
            case GgufTensorType.F32:
                Buffer.BlockCopy(tensor.RawData, 0, scale, 0, tensor.RawData.Length);
                break;
            case GgufTensorType.F16:
                GgufExtendedReader.DecodeFloat16(tensor.RawData, scale);
                break;
            default:
                throw new InvalidDataException(
                    $"Tensor '{key}' has unsupported type {tensor.GgmlType} for RmsNorm scale.");
        }
        norm.ImportScale(scale);
    }

    private static void ImportQ2_0IntoBitLinear(
        IReadOnlyDictionary<string, GgufRawTensor> byName, string key, BitLinear target)
    {
        if (!byName.TryGetValue(key, out var tensor))
        {
            throw new InvalidDataException($"GGUF missing tensor '{key}'.");
        }
        if (tensor.GgmlType != GgufTensorType.PrismQ2_0)
        {
            throw new InvalidDataException(
                $"Tensor '{key}' has type {tensor.GgmlType}; body weights must be Prism Q2_0.");
        }
        if (tensor.Dimensions.Count != 2)
        {
            throw new InvalidDataException(
                $"Tensor '{key}' has rank {tensor.Dimensions.Count}; body weights must be rank 2.");
        }

        // GGUF: ne[0] = innermost = input dim, ne[1] = outermost = output dim.
        int inDim = tensor.Dimensions[0];
        int outDim = tensor.Dimensions[1];

        if (inDim != target.Config.InputDimension || outDim != target.Config.OutputDimension)
        {
            throw new InvalidDataException(
                $"Tensor '{key}' shape [{inDim}, {outDim}] does not match target BitLinear "
                + $"[{target.Config.InputDimension}, {target.Config.OutputDimension}].");
        }

        int totalWeights = inDim * outDim;
        if (totalWeights % PrismQ2_0.BlockWeights != 0)
        {
            throw new InvalidDataException(
                $"Tensor '{key}' element count {totalWeights} is not a multiple of {PrismQ2_0.BlockWeights}.");
        }

        int blockCount = totalWeights / PrismQ2_0.BlockWeights;
        if (tensor.RawData.Length != blockCount * PrismQ2_0.BlockBytes)
        {
            throw new InvalidDataException(
                $"Tensor '{key}' byte length {tensor.RawData.Length} does not match "
                + $"{blockCount} blocks * {PrismQ2_0.BlockBytes} bytes.");
        }

        var trits = new sbyte[totalWeights];
        double weightedAbsSum = 0d;
        var raw = tensor.RawData.AsSpan();
        for (int b = 0; b < blockCount; b++)
        {
            var block = raw.Slice(b * PrismQ2_0.BlockBytes, PrismQ2_0.BlockBytes);
            var tritSpan = trits.AsSpan(b * PrismQ2_0.BlockWeights, PrismQ2_0.BlockWeights);
            PrismQ2_0.DecodeTritsInto(block, tritSpan, out double blockAbsSum);
            weightedAbsSum += blockAbsSum;
        }

        float gamma = (float)(weightedAbsSum / totalWeights);
        target.ImportTernary(trits, gamma);
    }

    private static string FormatDims(IReadOnlyList<int> dims) =>
        "[" + string.Join(", ", dims) + "]";
}
