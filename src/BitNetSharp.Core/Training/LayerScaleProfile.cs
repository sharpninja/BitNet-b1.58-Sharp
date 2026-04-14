namespace BitNetSharp.Core.Training;

/// <summary>
/// Per-layer calibration data for integer training.
/// Computed during the float calibration pass (Pass 1) and consumed
/// by the integer training pass (Pass 2).
/// </summary>
public sealed record LayerScaleProfile
{
    public required string LayerName { get; init; }
    public required int OutputDimension { get; init; }
    public required int InputDimension { get; init; }
    public required float MaxGradientMagnitude { get; init; }
    public required float P99GradientMagnitude { get; init; }
    public required float MeanGradientMagnitude { get; init; }
    public required float ObservedWeightRange { get; init; }
    public required float MaxWeightMagnitude { get; init; }
    public required float Epsilon { get; init; }
    public required int BucketCount { get; init; }

    public int TernaryThreshold => Epsilon > 0f ? (int)(0.5f / Epsilon) : 1;

    public static LayerScaleProfile Compute(
        string layerName,
        int outputDim,
        int inputDim,
        IReadOnlyList<float> gradientSamples,
        float minWeightSeen,
        float maxWeightSeen)
    {
        var sorted = gradientSamples.Select(MathF.Abs).OrderBy(static x => x).ToArray();
        var p99 = sorted.Length > 0 ? sorted[(int)(sorted.Length * 0.99)] : 1e-4f;
        var epsilon = MathF.Max(p99 / 32767f, 1e-9f);
        var weightRange = maxWeightSeen - minWeightSeen;
        var bucketSize = epsilon * 65536f;
        var bucketCount = (int)MathF.Ceiling(MathF.Max(weightRange, 1f) / bucketSize);

        return new LayerScaleProfile
        {
            LayerName = layerName,
            OutputDimension = outputDim,
            InputDimension = inputDim,
            MaxGradientMagnitude = sorted.Length > 0 ? sorted[^1] : 0f,
            P99GradientMagnitude = p99,
            MeanGradientMagnitude = sorted.Length > 0 ? sorted.Average() : 0f,
            ObservedWeightRange = weightRange,
            MaxWeightMagnitude = MathF.Max(MathF.Abs(minWeightSeen), MathF.Abs(maxWeightSeen)),
            Epsilon = epsilon,
            BucketCount = bucketCount
        };
    }
}
