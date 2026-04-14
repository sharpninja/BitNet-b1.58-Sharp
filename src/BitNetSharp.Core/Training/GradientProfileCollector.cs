using System.Security.Cryptography;
using System.Text;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Core.Training;

/// <summary>
/// Observes gradient and weight statistics during a float calibration pass
/// without interfering with the training loop. Produces a <see cref="ModelScaleProfile"/>
/// for use by the integer training pass.
/// </summary>
public sealed class GradientProfileCollector
{
    private readonly Dictionary<string, List<float>> _gradientSamples = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (float Min, float Max)> _weightRanges = new(StringComparer.Ordinal);

    public void RecordGradients(string layerName, float[] gradients)
    {
        ArgumentNullException.ThrowIfNull(gradients);

        if (!_gradientSamples.TryGetValue(layerName, out var samples))
        {
            _gradientSamples[layerName] = samples = [];
        }

        foreach (var g in gradients)
        {
            samples.Add(g);
        }
    }

    public void RecordWeights(string layerName, float[] weights)
    {
        ArgumentNullException.ThrowIfNull(weights);

        _weightRanges.TryGetValue(layerName, out var range);
        var min = range is { Min: 0f, Max: 0f } ? float.MaxValue : range.Min;
        var max = range is { Min: 0f, Max: 0f } ? float.MinValue : range.Max;

        foreach (var w in weights)
        {
            if (w < min) min = w;
            if (w > max) max = w;
        }

        _weightRanges[layerName] = (min, max);
    }

    public ModelScaleProfile BuildProfile(string modelId, BitNetConfig config, int calibrationSteps)
    {
        var layers = _gradientSamples.Keys
            .Select(name =>
            {
                var (min, max) = _weightRanges.GetValueOrDefault(name, (0f, 0f));
                return LayerScaleProfile.Compute(
                    name,
                    config.Dimension,
                    config.Dimension,
                    _gradientSamples[name],
                    min,
                    max);
            })
            .ToArray();

        return new ModelScaleProfile
        {
            ModelId = modelId,
            ArchitectureHash = ComputeArchitectureHash(config),
            CalibratedAt = DateTimeOffset.UtcNow,
            CalibrationSteps = calibrationSteps,
            Layers = layers
        };
    }

    private static string ComputeArchitectureHash(BitNetConfig config)
    {
        var repr = $"{config.VocabSize}:{config.Dimension}:{config.HiddenDimension}:" +
                   $"{config.LayerCount}:{config.HeadCount}:{config.MaxSequenceLength}";
        var bytes = SHA256.HashData(Encoding.UTF8.GetBytes(repr));
        return Convert.ToHexString(bytes)[..16];
    }
}
