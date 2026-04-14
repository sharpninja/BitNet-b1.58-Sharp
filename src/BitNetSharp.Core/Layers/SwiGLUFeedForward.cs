using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;
using BitNetSharp.Core.Utils;

namespace BitNetSharp.Core.Layers;

public sealed class SwiGLUFeedForward : Module
{
    public SwiGLUFeedForward(BitNetConfig config, Random random)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);

        Config = config;
        GateProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.HiddenDimension), random);
        UpProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.Dimension, config.HiddenDimension), random);
        DownProjection = ParameterInitializer.CreateBitLinear(new BitLinearConfig(config.HiddenDimension, config.Dimension), random);
    }

    public BitNetConfig Config { get; }

    public BitLinear GateProjection { get; }

    public BitLinear UpProjection { get; }

    public BitLinear DownProjection { get; }

    public bool UsesSwiGLUActivation => true;

    public long EstimateResidentParameterBytes() =>
        GateProjection.EstimateResidentParameterBytes()
        + UpProjection.EstimateResidentParameterBytes()
        + DownProjection.EstimateResidentParameterBytes();

    // Cached for backward pass
    private float[,]? _cachedGated;
    private float[,]? _cachedUp;
    private float[,]? _cachedActivated;

    public override float[,] Forward(float[,] input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var gated = GateProjection.Forward(input);
        var up = UpProjection.Forward(input);
        var activated = new float[gated.GetLength(0), gated.GetLength(1)];

        for (var row = 0; row < gated.GetLength(0); row++)
        {
            for (var column = 0; column < gated.GetLength(1); column++)
            {
                activated[row, column] = Silu(gated[row, column]);
            }
        }

        _cachedGated = gated;
        _cachedUp = up;
        _cachedActivated = activated;

        return DownProjection.Forward(TensorMath.ElementwiseMultiply(activated, up));
    }

    public override float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);

        if (_cachedGated is null || _cachedUp is null || _cachedActivated is null)
        {
            return (float[,])gradientOutput.Clone();
        }

        // Backward through DownProjection
        var gradDownInput = DownProjection.BackwardSTE(gradientOutput);

        var rows = gradDownInput.GetLength(0);
        var hiddenDim = Config.HiddenDimension;

        // gradDownInput = dL/d(activated * up)
        // dL/d_activated = gradDownInput * up
        // dL/d_up = gradDownInput * activated
        var gradActivated = new float[rows, hiddenDim];
        var gradUp = new float[rows, hiddenDim];

        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < hiddenDim; col++)
            {
                gradActivated[row, col] = gradDownInput[row, col] * _cachedUp[row, col];
                gradUp[row, col] = gradDownInput[row, col] * _cachedActivated[row, col];
            }
        }

        // dL/d_gated = dL/d_activated * SiLU'(gated)
        var gradGated = new float[rows, hiddenDim];
        for (var row = 0; row < rows; row++)
        {
            for (var col = 0; col < hiddenDim; col++)
            {
                gradGated[row, col] = gradActivated[row, col] * SiluDerivative(_cachedGated[row, col]);
            }
        }

        // Backward through GateProjection and UpProjection
        var gradInputFromGate = GateProjection.BackwardSTE(gradGated);
        var gradInputFromUp = UpProjection.BackwardSTE(gradUp);

        // Sum gradients from both paths (they share the same input)
        return TensorMath.Add(gradInputFromGate, gradInputFromUp);
    }

    private static float Silu(float value) => value / (1f + MathF.Exp(-value));

    private static float SiluDerivative(float value)
    {
        var sigmoid = 1f / (1f + MathF.Exp(-value));
        return sigmoid * (1f + value * (1f - sigmoid));
    }
}
