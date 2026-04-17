using System;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Configuration;
using Microsoft.Extensions.Options;

namespace BitNetSharp.Distributed.Coordinator.Services;

/// <summary>
/// Exposes the coordinator's chosen <see cref="BitNetConfig"/> and the
/// matching canonical flat-parameter vector length (per
/// <see cref="FlatParameterPack.ComputeLength"/>). Resolved once at
/// startup from <see cref="CoordinatorOptions.ModelPreset"/> so the
/// weight service, gradient endpoint, and any other consumer agree on
/// the shape of the weight vector a worker sees.
/// </summary>
public interface ICoordinatorModelConfig
{
    /// <summary>Preset name that selected the config (e.g. "small").</summary>
    string PresetName { get; }

    /// <summary>BitNet architecture config the preset resolves to.</summary>
    BitNetConfig Config { get; }

    /// <summary>
    /// Canonical flat-parameter vector length for <see cref="Config"/>
    /// as computed by <see cref="FlatParameterPack.ComputeLength"/>.
    /// Both the in-memory weight vector and every accepted gradient
    /// submission must be exactly this many fp32 elements.
    /// </summary>
    int FlatLength { get; }

    /// <summary>Human-readable summary for banners + logs.</summary>
    string ToDisplayString();
}

/// <summary>
/// Default <see cref="ICoordinatorModelConfig"/> built from the
/// configured preset at service construction. The compute
/// (<see cref="FlatParameterPack.ComputeLength"/>) runs exactly once,
/// so every consumer reads cached values.
/// </summary>
public sealed class CoordinatorModelConfig : ICoordinatorModelConfig
{
    public CoordinatorModelConfig(IOptions<CoordinatorOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var presetName = string.IsNullOrWhiteSpace(options.Value.ModelPreset)
            ? "small"
            : options.Value.ModelPreset!;

        var preset = TruckMateModelPresets.GetPreset(presetName);
        PresetName = presetName;
        Config = new BitNetConfig(
            vocabSize: preset.VocabSize,
            dimension: preset.Dimension,
            hiddenDimension: preset.HiddenDimension,
            layerCount: preset.LayerCount,
            headCount: preset.HeadCount,
            maxSequenceLength: preset.MaxSequenceLength);
        FlatLength = FlatParameterPack.ComputeLength(Config);
    }

    public string PresetName { get; }
    public BitNetConfig Config { get; }
    public int FlatLength { get; }

    public string ToDisplayString() =>
        $"preset={PresetName} vocab={Config.VocabSize} dim={Config.Dimension} "
        + $"hidden={Config.HiddenDimension} layers={Config.LayerCount} heads={Config.HeadCount} "
        + $"seq={Config.MaxSequenceLength} flatLength={FlatLength} "
        + $"bytesOnWire={WeightBlobCodec.HeaderSize + 4L * FlatLength}";
}
