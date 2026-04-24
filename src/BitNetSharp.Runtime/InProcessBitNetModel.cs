using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Runtime;

/// <summary>
/// Result returned by <see cref="InProcessBitNetModel.GenerateResponse"/>.
/// Structurally identical to the App-side <c>HostedAgentModelResponse</c>
/// but Runtime is AOT-only and has no App dependency.
/// </summary>
public sealed record InProcessResponse(
    string Text,
    IReadOnlyList<string> Diagnostics);

/// <summary>
/// AOT-compatible in-process BitNet inference engine. Loads a
/// <see cref="WeightBlobCodec"/> v1 (fp32) blob from disk and feeds its
/// weights into a freshly-constructed <see cref="BitNetPaperModel"/>
/// via <see cref="FlatParameterPack.Unpack"/>.
///
/// <para>
/// Phase 5 (refactored-rossum plan) will add v2 ternary-packed blobs;
/// this class remains the load/inference path for both formats.
/// </para>
/// </summary>
public sealed class InProcessBitNetModel : IDisposable
{
    private readonly BitNetPaperModel _model;
    private bool _disposed;

    /// <summary>Weight-blob version that was loaded (from the blob header).</summary>
    public long WeightVersion { get; }

    /// <summary>Underlying paper-aligned model; exposed for inspection + tests.</summary>
    public BitNetPaperModel Model => _model;

    /// <summary>Primary language inherited from <see cref="BitNetOptions"/>.</summary>
    public string PrimaryLanguage => _model.Options.PrimaryLanguage;

    /// <summary>Verbosity inherited from <see cref="BitNetOptions"/>.</summary>
    public VerbosityLevel Verbosity => _model.Options.Verbosity;

    /// <summary>Stable model identifier (delegates to <see cref="BitNetPaperModel.ModelId"/>).</summary>
    public string ModelId => _model.ModelId;

    private InProcessBitNetModel(BitNetPaperModel model, long weightVersion)
    {
        _model = model;
        WeightVersion = weightVersion;
    }

    /// <summary>
    /// Async factory: reads <paramref name="weightBlobPath"/>, validates the header,
    /// and returns a populated <see cref="InProcessBitNetModel"/>.
    /// Throws <see cref="ArgumentException"/> on bad magic / malformed payload.
    /// </summary>
    public static async Task<InProcessBitNetModel> LoadAsync(
        string weightBlobPath,
        BitNetOptions options,
        BitNetConfig? config = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(weightBlobPath);
        ArgumentNullException.ThrowIfNull(options);

        var (version, weights) = await WeightBlobCodec.DecodeAsync(weightBlobPath, cancellationToken)
            .ConfigureAwait(false);

        return CreateFromWeights(version, weights, options, config);
    }

    /// <summary>
    /// Synchronous factory for in-memory payloads (tests, embedded resources).
    /// Validates header then hydrates a <see cref="BitNetPaperModel"/>.
    /// </summary>
    public static InProcessBitNetModel LoadFromBytes(
        ReadOnlySpan<byte> payload,
        BitNetOptions options,
        BitNetConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(options);

        var weights = WeightBlobCodec.Decode(payload, out var version);
        return CreateFromWeights(version, weights, options, config);
    }

    private static InProcessBitNetModel CreateFromWeights(
        long version,
        float[] weights,
        BitNetOptions options,
        BitNetConfig? config)
    {
        var resolvedConfig = config ?? new BitNetConfig();
        var expectedLength = FlatParameterPack.ComputeLength(resolvedConfig);
        if (weights.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Weight blob has {weights.Length} floats but the configured transformer expects "
                + $"{expectedLength}. Config mismatch between coordinator training run and "
                + $"runtime BitNetConfig? Vocab={resolvedConfig.VocabSize} Layers={resolvedConfig.LayerCount} "
                + $"Dim={resolvedConfig.Dimension}.",
                nameof(weights));
        }

        var model = new BitNetPaperModel(options, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, resolvedConfig);
        FlatParameterPack.Unpack(model.Transformer, weights);
        return new InProcessBitNetModel(model, version);
    }

    /// <summary>Generate a response for a single prompt.</summary>
    public InProcessResponse GenerateResponse(string prompt, int? maxOutputTokens = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var result = _model.GenerateResponse(prompt, maxOutputTokens);
        return new InProcessResponse(result.ResponseText, result.Diagnostics);
    }

    /// <summary>Expose ternary-weight stats for diagnostics and Phase 5 promotion gating.</summary>
    public TernaryWeightStats GetTernaryWeightStats() => _model.GetTernaryWeightStats();

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
    }
}
