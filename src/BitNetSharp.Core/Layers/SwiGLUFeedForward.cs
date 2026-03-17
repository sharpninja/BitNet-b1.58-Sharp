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

        return DownProjection.Forward(TensorMath.ElementwiseMultiply(activated, up));
    }

    private static float Silu(float value) => value / (1f + MathF.Exp(-value));
}
