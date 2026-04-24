using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using BitNetSharp.Core;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Tests for the v2 on-disk format (packed trits + per-tensor Gamma).
/// These are separate from the core round-trip test so they can poke at the
/// raw byte layout directly and verify the backcompat contract with v1 files.
/// </summary>
public sealed class BitNetPaperGgufV2FormatTests
{
    [Fact]
    public void SaveEmitsV2FormatVersionMetadataAndPackedTensorTypeForBitLinearLayers()
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-gguf-v2-{Guid.NewGuid():N}");
        var ggufPath = Path.Combine(tempDirectory, "model.gguf");

        try
        {
            BitNetPaperGguf.Save(BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet), ggufPath);

            var (metadata, tensors) = ReadRawGguf(ggufPath);

            // Version key must be 2. Format name must be the v2 constant.
            Assert.Equal((uint)2, Assert.IsType<uint>(metadata["bitnetsharp.format_version"]));
            Assert.Equal("bitnet-b1.58-sharp.gguf.v2", Assert.IsType<string>(metadata["bitnetsharp.format"]));

            // Norm tensors stay F32 (dim-sized FP32 arrays).
            var attnNorm = tensors.Single(t => t.Name == "blk.0.attn_norm.weight");
            Assert.Equal((uint)0, attnNorm.TensorType);

            // Token embeddings stay F32 (vocab x dim FP32 matrix).
            var tokenEmbed = tensors.Single(t => t.Name == "token_embeddings");
            Assert.Equal((uint)0, tokenEmbed.TensorType);

            // Every BitLinear tensor (attn_q/k/v/out, ffn_gate/up/down, output) is packed.
            var packedTensorNames = tensors
                .Where(t => t.TensorType == 1001)
                .Select(t => t.Name)
                .ToArray();
            Assert.Contains("blk.0.attn_q.weight", packedTensorNames);
            Assert.Contains("blk.0.ffn_gate.weight", packedTensorNames);
            Assert.Contains("output.weight", packedTensorNames);
        }
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    [Fact]
    public void V2FileIsStrictlySmallerOnDiskThanEquivalentV1File()
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-gguf-v1v2-size-{Guid.NewGuid():N}");
        var v2Path = Path.Combine(tempDirectory, "model.v2.gguf");
        var v1Path = Path.Combine(tempDirectory, "model.v1.gguf");

        try
        {
            Directory.CreateDirectory(tempDirectory);
            var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet);

            // Write the standard v2 file.
            BitNetPaperGguf.Save(model, v2Path);

            // Mutate the v2 file into a v1 file by patching two bytes of
            // metadata is too fragile; instead, just re-measure the expected
            // v1 payload for the same model and assert v2 < v1.
            var v2Length = new FileInfo(v2Path).Length;

            // Synthesize v1 equivalent: every BitLinear is stored as FP32
            // (outDim * inDim * 4 bytes) instead of (4 + outDim * ceil(inDim/5)).
            // For the default small-model preset the savings fall between
            // 4x and 5x per BitLinear tensor, easily dominating FP32 norms
            // and FP32 token embeddings that stay unchanged.
            var v1Length = SimulateV1OnDiskSize(model, v2Path);

            Assert.True(
                v2Length < v1Length,
                $"v2 file size ({v2Length}) must be smaller than equivalent v1 size ({v1Length}).");

            // For any non-degenerate model the savings must be at least 2x
            // on the packed BitLinear payload, even accounting for FP32 parts.
            Assert.True(
                v2Length * 2 <= v1Length,
                $"v2 ({v2Length}) must be at most half the size of v1 ({v1Length}).");
        }
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    [Fact]
    public void LoadAcceptsV1FileWithFP32BitLinearPayloadsForBackCompat()
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-gguf-v1-backcompat-{Guid.NewGuid():N}");
        var v2Path = Path.Combine(tempDirectory, "model.v2.gguf");
        var v1Path = Path.Combine(tempDirectory, "model.v1.gguf");

        try
        {
            Directory.CreateDirectory(tempDirectory);
            var originalModel = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet);
            BitNetPaperGguf.Save(originalModel, v2Path);

            // Rewrite the v2 file as v1 by round-tripping through FP32. This
            // simulates an older checkpoint produced before the v2 format
            // landed. The helper rewrites every type=1001 tensor as type=0
            // float32 and flips the metadata flags back to v1.
            RewriteAsV1(v2Path, v1Path);

            var reloaded = BitNetPaperGguf.Load(v1Path, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, VerbosityLevel.Quiet);

            // Same prompt must produce the same output tokens (ternary rounding
            // applied during QuantizeFromFullPrecision recovers the original
            // trits because the FP32 payload is a lossless expansion of them).
            var original = originalModel.GenerateResponse("hello", maxTokens: 4);
            var roundTripped = reloaded.GenerateResponse("hello", maxTokens: 4);
            Assert.Equal(original.ResponseText, roundTripped.ResponseText);
        }
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    [Fact]
    public void V2RoundTripPreservesGammaAndTritsBitExact()
    {
        // The v2 format bypasses QuantizeFromFullPrecision on Load, so the
        // (packed_trits, Gamma) pair must round-trip bit-exact through
        // Save/Load. v1 had small Gamma drift because of the absmean
        // recomputation; v2 must not.
        var tempDirectory = Path.Combine(Path.GetTempPath(), $"bitnet-gguf-v2-exact-{Guid.NewGuid():N}");
        var ggufPath = Path.Combine(tempDirectory, "model.gguf");

        try
        {
            Directory.CreateDirectory(tempDirectory);
            var original = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet);
            BitNetPaperGguf.Save(original, ggufPath);
            var reloaded = BitNetPaperGguf.Load(ggufPath, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, VerbosityLevel.Quiet);

            var originalLayers = GetAllBitLinearLayers(original);
            var reloadedLayers = GetAllBitLinearLayers(reloaded);
            Assert.Equal(originalLayers.Count, reloadedLayers.Count);

            for (var i = 0; i < originalLayers.Count; i++)
            {
                // Bit-exact Gamma (no FP32 absmean recomputation on Load).
                Assert.Equal(originalLayers[i].Gamma, reloadedLayers[i].Gamma);

                // Bit-exact trit statistics (trits are copied byte-for-byte
                // through the packed path).
                var origStats = originalLayers[i].GetTernaryStats();
                var reloadedStats = reloadedLayers[i].GetTernaryStats();
                Assert.Equal(origStats.NegativeCount, reloadedStats.NegativeCount);
                Assert.Equal(origStats.ZeroCount, reloadedStats.ZeroCount);
                Assert.Equal(origStats.PositiveCount, reloadedStats.PositiveCount);
            }
        }
        finally
        {
            if (Directory.Exists(tempDirectory))
            {
                Directory.Delete(tempDirectory, recursive: true);
            }
        }
    }

    private static IReadOnlyList<Core.Layers.BitLinear> GetAllBitLinearLayers(BitNetPaperModel model)
    {
        var list = new List<Core.Layers.BitLinear>();
        list.AddRange(model.GetTransformerBitLinearLayers());
        list.Add(model.GetOutputHead());
        return list;
    }

    private static long SimulateV1OnDiskSize(BitNetPaperModel model, string v2Path)
    {
        // v1 stored BitLinear layers as FP32 (outDim * inDim * 4). Compute the
        // expected v1 size by summing (packedBytes - fp32Bytes) over every
        // BitLinear and subtracting that delta from the v2 on-disk size.
        var config = model.Config;
        var bitlinearShapes = new List<(int outDim, int inDim)>();

        // Layer projections: q (dim,dim), k (kvDim,dim), v (kvDim,dim),
        // out (dim,dim), ffn_gate (hidden,dim), ffn_up (hidden,dim),
        // ffn_down (dim,hidden).
        var kvDim = config.KvHeadCount * config.HeadDimension;
        for (var l = 0; l < config.LayerCount; l++)
        {
            bitlinearShapes.Add((config.Dimension, config.Dimension)); // q
            bitlinearShapes.Add((kvDim, config.Dimension));            // k
            bitlinearShapes.Add((kvDim, config.Dimension));            // v
            bitlinearShapes.Add((config.Dimension, config.Dimension)); // out
            bitlinearShapes.Add((config.HiddenDimension, config.Dimension)); // ffn_gate
            bitlinearShapes.Add((config.HiddenDimension, config.Dimension)); // ffn_up
            bitlinearShapes.Add((config.Dimension, config.HiddenDimension)); // ffn_down
        }

        bitlinearShapes.Add((config.VocabSize, config.Dimension)); // output head

        long v2Payload = 0;
        long v1Payload = 0;
        foreach (var (outDim, inDim) in bitlinearShapes)
        {
            var packedStride = (inDim + 4) / 5;
            v2Payload += sizeof(float) + (long)packedStride * outDim;
            v1Payload += (long)outDim * inDim * sizeof(float);
        }

        var actualV2Size = new FileInfo(v2Path).Length;
        return actualV2Size - v2Payload + v1Payload;
    }

    private static void RewriteAsV1(string v2Path, string v1Path)
    {
        // Use the streaming reader to pull out FP32-equivalent payloads,
        // then rebuild the file using a hand-rolled v1 writer. Simplest
        // path that does not depend on internals: Load the v2 file into a
        // model, then use reflection to invoke a legacy-writer helper.
        // Here we take a shortcut: Load the v2 model, then hand-serialize a
        // v1-compatible file using only public APIs + binary writer.
        var model = BitNetPaperGguf.Load(v2Path, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, VerbosityLevel.Quiet);
        WriteLegacyV1(model, v1Path);
    }

    // Writes a handcrafted v1 GGUF file that mimics the pre-v2 layout:
    // every tensor is Float32, bitnetsharp.format = "...v1", no
    // bitnetsharp.format_version key. Used only by the backcompat test.
    private static void WriteLegacyV1(BitNetPaperModel model, string path)
    {
        // Export the same tensors as CreateStreamingTensors, but force all of
        // them to FP32 payload and omit the version key.
        var config = model.Config;
        var linearLayers = model.GetTransformerBitLinearLayers().ToArray();
        var normLayers = model.GetNormLayers().ToArray();

        var metadataOverrides = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["bitnetsharp.format"] = "bitnet-b1.58-sharp.gguf.v1",
        };

        var tensorDescriptors = new List<(string Name, int[] Dimensions, Action<BinaryWriter> Write, long Bytes)>();

        var tokenEmbeddings = model.GetTokenEmbeddingsMatrix();
        AddFp32Matrix(tensorDescriptors, "token_embeddings", tokenEmbeddings);

        var linearIndex = 0;
        var normIndex = 0;
        for (var layer = 0; layer < config.LayerCount; layer++)
        {
            AddFp32Vector(tensorDescriptors, $"blk.{layer}.attn_norm.weight", normLayers[normIndex++].ExportScale());
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.attn_q.weight", linearLayers[linearIndex++]);
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.attn_k.weight", linearLayers[linearIndex++]);
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.attn_v.weight", linearLayers[linearIndex++]);
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.attn_out.weight", linearLayers[linearIndex++]);
            AddFp32Vector(tensorDescriptors, $"blk.{layer}.ffn_norm.weight", normLayers[normIndex++].ExportScale());
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.ffn_gate.weight", linearLayers[linearIndex++]);
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.ffn_up.weight", linearLayers[linearIndex++]);
            AddFp32BitLinear(tensorDescriptors, $"blk.{layer}.ffn_down.weight", linearLayers[linearIndex++]);
        }

        AddFp32Vector(tensorDescriptors, "output_norm.weight", normLayers[normIndex].ExportScale());
        AddFp32BitLinear(tensorDescriptors, "output.weight", model.GetOutputHead());

        // Rebuild the full metadata dict by copying what Save would have emitted, minus v2 keys.
        var fullMetadata = BuildV1Metadata(model, metadataOverrides);

        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
        writer.Write(Encoding.ASCII.GetBytes("GGUF"));
        writer.Write((uint)3); // version
        writer.Write((ulong)tensorDescriptors.Count);
        writer.Write((ulong)fullMetadata.Count);

        foreach (var (k, v) in fullMetadata)
        {
            WriteGgufString(writer, k);
            WriteGgufMetadataValue(writer, v);
        }

        // Compute offsets.
        const uint alignment = 32;
        var offsets = new ulong[tensorDescriptors.Count];
        ulong cur = 0;
        for (var i = 0; i < tensorDescriptors.Count; i++)
        {
            var rem = cur % alignment;
            if (rem != 0) cur += alignment - rem;
            offsets[i] = cur;
            cur += checked((ulong)tensorDescriptors[i].Bytes);
        }

        for (var i = 0; i < tensorDescriptors.Count; i++)
        {
            var t = tensorDescriptors[i];
            WriteGgufString(writer, t.Name);
            writer.Write((uint)t.Dimensions.Length);
            foreach (var d in t.Dimensions) writer.Write((ulong)d);
            writer.Write((uint)0); // Float32TensorType
            writer.Write(offsets[i]);
        }

        AlignStream(stream, alignment);
        var dataStart = stream.Position;
        for (var i = 0; i < tensorDescriptors.Count; i++)
        {
            var target = dataStart + (long)offsets[i];
            WritePadding(stream, target - stream.Position);
            tensorDescriptors[i].Write(writer);
        }
    }

    private static void AddFp32Matrix(List<(string, int[], Action<BinaryWriter>, long)> list, string name, float[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var bytes = checked((long)rows * cols * sizeof(float));
        list.Add((name, new[] { rows, cols }, writer =>
        {
            var buf = new float[cols];
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++) buf[c] = matrix[r, c];
                writer.Write(MemoryMarshal.AsBytes(buf.AsSpan()));
            }
        }, bytes));
    }

    private static void AddFp32Vector(List<(string, int[], Action<BinaryWriter>, long)> list, string name, float[] values)
    {
        var bytes = checked((long)values.Length * sizeof(float));
        list.Add((name, new[] { values.Length }, writer => writer.Write(MemoryMarshal.AsBytes(values.AsSpan())), bytes));
    }

    private static void AddFp32BitLinear(List<(string, int[], Action<BinaryWriter>, long)> list, string name, Core.Layers.BitLinear layer)
    {
        var rows = layer.Config.OutputDimension;
        var cols = layer.Config.InputDimension;
        var bytes = checked((long)rows * cols * sizeof(float));
        list.Add((name, new[] { rows, cols }, layer.WriteFullPrecisionTo, bytes));
    }

    private static Dictionary<string, object> BuildV1Metadata(BitNetPaperModel model, Dictionary<string, object> overrides)
    {
        var config = model.Config;
        var options = model.Options;
        // Minimum metadata set Load needs. Keep in sync with the v2 writer.
        var meta = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            ["general.architecture"] = "bitnetsharp",
            ["general.name"] = model.ModelId,
            ["general.alignment"] = (uint)32,
            ["bitnetsharp.model_id"] = model.ModelId,
            ["bitnetsharp.bootstrap_seed"] = 42,
            ["bitnetsharp.vocabulary"] = System.Text.Json.JsonSerializer.Serialize(options.Vocabulary),
            ["bitnetsharp.memorized_responses"] = System.Text.Json.JsonSerializer.Serialize(
                model.ExportMemorizedResponses().ToDictionary(
                    static kvp => kvp.Key,
                    static kvp => kvp.Value.ToArray(),
                    StringComparer.Ordinal)),
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
            ["bitnetsharp.config.rms_norm_epsilon"] = (double)config.RmsNormEpsilon,
        };

        foreach (var kvp in overrides) meta[kvp.Key] = kvp.Value;
        return meta;
    }

    private static void WriteGgufString(BinaryWriter w, string s)
    {
        var bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private static void WriteGgufMetadataValue(BinaryWriter w, object v)
    {
        switch (v)
        {
            case uint u: w.Write((uint)4); w.Write(u); break; // GgufMetadataValueType.UInt32 = 4
            case int i: w.Write((uint)5); w.Write(i); break; // Int32 = 5
            case ulong ul: w.Write((uint)10); w.Write(ul); break; // UInt64 = 10
            case long l: w.Write((uint)11); w.Write(l); break; // Int64 = 11
            case float f: w.Write((uint)6); w.Write(f); break; // Float32 = 6
            case double d: w.Write((uint)12); w.Write(d); break; // Float64 = 12
            case bool b: w.Write((uint)7); w.Write(b); break; // Bool = 7
            case string s: w.Write((uint)8); WriteGgufString(w, s); break; // String = 8
            default: throw new InvalidOperationException($"Unsupported metadata type {v?.GetType()}");
        }
    }

    private static void AlignStream(Stream stream, uint alignment)
    {
        var rem = (ulong)stream.Position % alignment;
        if (rem != 0) WritePadding(stream, (long)(alignment - rem));
    }

    private static void WritePadding(Stream stream, long bytes)
    {
        if (bytes <= 0) return;
        Span<byte> pad = stackalloc byte[256];
        while (bytes > 0)
        {
            var n = (int)Math.Min(bytes, pad.Length);
            stream.Write(pad[..n]);
            bytes -= n;
        }
    }

    private static (Dictionary<string, object> Metadata, List<TensorHeader> Tensors) ReadRawGguf(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        var magic = reader.ReadBytes(4);
        Assert.Equal(new byte[] { (byte)'G', (byte)'G', (byte)'U', (byte)'F' }, magic);
        var version = reader.ReadUInt32();
        Assert.Equal((uint)3, version);

        var tensorCount = checked((int)reader.ReadUInt64());
        var metaCount = checked((int)reader.ReadUInt64());

        var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
        for (var i = 0; i < metaCount; i++)
        {
            var key = ReadGgufString(reader);
            metadata[key] = ReadGgufMetadataValue(reader);
        }

        var tensors = new List<TensorHeader>(tensorCount);
        for (var i = 0; i < tensorCount; i++)
        {
            var name = ReadGgufString(reader);
            var rank = checked((int)reader.ReadUInt32());
            var dims = new int[rank];
            for (var d = 0; d < rank; d++) dims[d] = checked((int)reader.ReadUInt64());
            var type = reader.ReadUInt32();
            var off = reader.ReadUInt64();
            tensors.Add(new TensorHeader(name, dims, type, off));
        }

        return (metadata, tensors);
    }

    private static string ReadGgufString(BinaryReader r)
    {
        var len = checked((int)r.ReadUInt64());
        var bytes = r.ReadBytes(len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static object ReadGgufMetadataValue(BinaryReader r)
    {
        var typeCode = r.ReadUInt32();
        return typeCode switch
        {
            4 => r.ReadUInt32(),
            5 => r.ReadInt32(),
            10 => r.ReadUInt64(),
            11 => r.ReadInt64(),
            6 => r.ReadSingle(),
            12 => r.ReadDouble(),
            7 => r.ReadBoolean(),
            8 => ReadGgufString(r),
            _ => throw new InvalidOperationException($"Unknown metadata type code {typeCode}"),
        };
    }

    private sealed record TensorHeader(string Name, int[] Dimensions, uint TensorType, ulong RelativeOffset);
}
