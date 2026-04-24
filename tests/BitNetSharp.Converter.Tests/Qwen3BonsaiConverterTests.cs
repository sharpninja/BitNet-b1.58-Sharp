using System;
using System.Collections.Generic;
using System.IO;
using BitNetSharp.Core;
using BitNetSharp.Core.Converters;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Serialization.Gguf;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Converter.Tests;

public sealed class Qwen3BonsaiConverterTests
{
    // Mini shape matching Qwen3 proportions: dim=64, headCount=4, kvHeadCount=2,
    // headDim=16, hidden=128, layers=2.
    private const int Dim = 64;
    private const int HeadCount = 4;
    private const int KvHeadCount = 2;
    private const int HeadDim = 16;
    private const int KvDim = KvHeadCount * HeadDim;
    private const int Hidden = 128;
    private const int LayerCount = 2;

    [Fact]
    public void Import_RejectsGgufWithWrongArchitecture()
    {
        var doc = BuildSyntheticGguf("llama");
        var target = BuildMiniModel();

        var ex = Assert.Throws<InvalidDataException>(() => Qwen3BonsaiConverter.Import(doc, target));
        Assert.Contains("qwen3", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Import_RejectsWhenArchitectureMetadataMissing()
    {
        var doc = BuildSyntheticGgufWithoutArchitecture();
        var target = BuildMiniModel();

        Assert.Throws<InvalidDataException>(() => Qwen3BonsaiConverter.Import(doc, target));
    }

    [Fact]
    public void Import_PopulatesBitLinearGammaForEveryBodyTensor()
    {
        var doc = BuildSyntheticGguf("qwen3");
        var target = BuildMiniModel();

        Qwen3BonsaiConverter.Import(doc, target);

        for (int i = 0; i < LayerCount; i++)
        {
            var layer = target.Transformer.Layers[i];
            Assert.True(layer.Attention.QueryProjection.Gamma > 0f,
                $"layer {i} Q gamma should be > 0");
            Assert.True(layer.Attention.KeyProjection.Gamma > 0f,
                $"layer {i} K gamma should be > 0");
            Assert.True(layer.Attention.ValueProjection.Gamma > 0f,
                $"layer {i} V gamma should be > 0");
            Assert.True(layer.Attention.OutputProjection.Gamma > 0f,
                $"layer {i} O gamma should be > 0");
            Assert.True(layer.FeedForward.GateProjection.Gamma > 0f,
                $"layer {i} gate gamma should be > 0");
            Assert.True(layer.FeedForward.UpProjection.Gamma > 0f,
                $"layer {i} up gamma should be > 0");
            Assert.True(layer.FeedForward.DownProjection.Gamma > 0f,
                $"layer {i} down gamma should be > 0");
        }
    }

    [Fact]
    public void Import_ComputedGamma_EqualsAnalyticFormulaForKnownCodes()
    {
        // With every Q2_0 byte = 0xE4 (q = 0,1,2,3) and d=0.5 per block:
        // per block: count_q0=32, count_q2=32, count_q3=32, count_q1=32
        // weightedAbsSum = 0.5 * (32 + 32 + 2*32) = 64
        // total_weights = 128
        // Gamma = 64/128 = 0.5
        var doc = BuildSyntheticGguf("qwen3", q2Scale: 0.5f, q2Pattern: 0xE4);
        var target = BuildMiniModel();

        Qwen3BonsaiConverter.Import(doc, target);

        var q = target.Transformer.Layers[0].Attention.QueryProjection;
        Assert.Equal(0.5f, q.Gamma, 3);
    }

    [Fact]
    public void Import_SetsRmsNormScaleFromGguf()
    {
        var doc = BuildSyntheticGguf("qwen3", normScaleValue: 1.25f);
        var target = BuildMiniModel();

        Qwen3BonsaiConverter.Import(doc, target);

        for (int i = 0; i < LayerCount; i++)
        {
            var preAttn = target.Transformer.Layers[i].PreAttentionNorm.ExportScale();
            var preFfn = target.Transformer.Layers[i].PreFeedForwardNorm.ExportScale();
            Assert.All(preAttn, v => Assert.Equal(1.25f, v, 2));
            Assert.All(preFfn, v => Assert.Equal(1.25f, v, 2));
        }

        var finalScale = target.Transformer.FinalNorm.ExportScale();
        Assert.All(finalScale, v => Assert.Equal(1.25f, v, 2));
    }

    [Fact]
    public void Import_RejectsWhenLayerCountMismatches()
    {
        // GGUF has LayerCount blocks; target model has LayerCount+1.
        var doc = BuildSyntheticGguf("qwen3");
        var target = BuildMiniModel(overrideLayerCount: LayerCount + 1);

        Assert.Throws<InvalidDataException>(() => Qwen3BonsaiConverter.Import(doc, target));
    }

    [Fact]
    public void Import_DiscardsTokenEmbeddingAndOutputTensors()
    {
        // token_embd and output are discarded even if present in the GGUF.
        // Bootstrap-seeded embeddings stay at their constructor-initialized
        // random values (not overwritten).
        var doc = BuildSyntheticGguf("qwen3");
        var target = BuildMiniModel();

        // Capture a single embedding value before import
        float before = ReadFirstEmbeddingValue(target);

        Qwen3BonsaiConverter.Import(doc, target);

        float after = ReadFirstEmbeddingValue(target);
        Assert.Equal(before, after);
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static BitNetPaperModel BuildMiniModel(int? overrideLayerCount = null)
    {
        var options = new BitNetOptions(
            Vocabulary: new[] { "the", "a", "cat", "dog", "runs" },
            Verbosity: VerbosityLevel.Quiet);

        int layers = overrideLayerCount ?? LayerCount;
        var config = new BitNetConfig(
            vocabSize: 8, // Begin, End, Unk + 5 tokens
            dimension: Dim,
            hiddenDimension: Hidden,
            layerCount: layers,
            headCount: HeadCount,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: KvHeadCount,
            ropeTheta: 10_000f);

        return new BitNetPaperModel(options, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, config, seed: 1);
    }

    private static float ReadFirstEmbeddingValue(BitNetPaperModel model)
    {
        // Use the public forward path: embed token id 0 and read first dim of output.
        // This indirectly reads _tokenEmbeddings[0, 0] before RmsNorm/attention.
        // Simpler: use ExportTokenEmbeddings via the Transformer. But that's
        // internal — our Converter tests have InternalsVisibleTo, so use it.
        var emb = model.Transformer.ExportTokenEmbeddings();
        return emb[0, 0];
    }

    private static GgufExtendedDocument BuildSyntheticGgufWithoutArchitecture()
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.alignment"] = 32u,
            ["qwen3.block_count"] = (uint)LayerCount,
        };
        var tensors = BuildBodyTensors(0.5f, 0xE4, 1.25f);
        byte[] bytes = SyntheticGguf.Build(metadata, tensors);
        using var stream = new MemoryStream(bytes);
        return GgufExtendedReader.Read(stream);
    }

    private static GgufExtendedDocument BuildSyntheticGguf(
        string architecture,
        float q2Scale = 0.5f,
        byte q2Pattern = 0xE4,
        float normScaleValue = 1.25f)
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.alignment"] = 32u,
            ["general.architecture"] = architecture,
            ["qwen3.block_count"] = (uint)LayerCount,
            ["qwen3.embedding_length"] = (uint)Dim,
            ["qwen3.attention.head_count"] = (uint)HeadCount,
            ["qwen3.attention.head_count_kv"] = (uint)KvHeadCount,
            ["qwen3.feed_forward_length"] = (uint)Hidden,
        };
        var tensors = BuildBodyTensors(q2Scale, q2Pattern, normScaleValue);
        byte[] bytes = SyntheticGguf.Build(metadata, tensors);
        using var stream = new MemoryStream(bytes);
        return GgufExtendedReader.Read(stream);
    }

    private static List<SyntheticGguf.Tensor> BuildBodyTensors(float q2Scale, byte q2Pattern, float normScaleValue)
    {
        var tensors = new List<SyntheticGguf.Tensor>();

        // output_norm.weight
        tensors.Add(new SyntheticGguf.Tensor(
            "output_norm.weight", new[] { Dim }, GgufTensorType.F16, BuildF16NormScale(Dim, normScaleValue)));

        // token_embd.weight and output.weight: include but the converter should discard them
        tensors.Add(new SyntheticGguf.Tensor(
            "token_embd.weight", new[] { Dim, 8 }, GgufTensorType.F16, new byte[Dim * 8 * 2]));
        tensors.Add(new SyntheticGguf.Tensor(
            "output.weight", new[] { Dim, 8 }, GgufTensorType.F16, new byte[Dim * 8 * 2]));

        for (int i = 0; i < LayerCount; i++)
        {
            string p = $"blk.{i}.";
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_norm.weight", new[] { Dim }, GgufTensorType.F16, BuildF16NormScale(Dim, normScaleValue)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_q.weight", new[] { Dim, Dim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Dim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_k.weight", new[] { Dim, KvDim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * KvDim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_v.weight", new[] { Dim, KvDim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * KvDim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "attn_output.weight", new[] { Dim, Dim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Dim, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_norm.weight", new[] { Dim }, GgufTensorType.F16, BuildF16NormScale(Dim, normScaleValue)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_gate.weight", new[] { Dim, Hidden }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Hidden, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_up.weight", new[] { Dim, Hidden }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Dim * Hidden, q2Scale, q2Pattern)));
            tensors.Add(new SyntheticGguf.Tensor(
                p + "ffn_down.weight", new[] { Hidden, Dim }, GgufTensorType.PrismQ2_0, BuildQ2Bytes(Hidden * Dim, q2Scale, q2Pattern)));
        }

        return tensors;
    }

    private static byte[] BuildF16NormScale(int dim, float value)
    {
        byte[] bytes = new byte[dim * 2];
        ushort halfBits = BitConverter.HalfToUInt16Bits((Half)value);
        for (int i = 0; i < dim; i++)
        {
            bytes[i * 2] = (byte)(halfBits & 0xFF);
            bytes[i * 2 + 1] = (byte)((halfBits >> 8) & 0xFF);
        }
        return bytes;
    }

    private static byte[] BuildQ2Bytes(int elementCount, float scale, byte codePattern)
    {
        int blocks = elementCount / 128;
        byte[] bytes = new byte[blocks * 34];
        ushort scaleBits = BitConverter.HalfToUInt16Bits((Half)scale);
        for (int b = 0; b < blocks; b++)
        {
            int off = b * 34;
            bytes[off] = (byte)(scaleBits & 0xFF);
            bytes[off + 1] = (byte)((scaleBits >> 8) & 0xFF);
            for (int j = 0; j < 32; j++)
            {
                bytes[off + 2 + j] = codePattern;
            }
        }
        return bytes;
    }
}
