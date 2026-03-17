namespace BitNetSharp.Core.Quantization;

/// <summary>
/// Configuration for a BitLinear layer using bias-free projections.
/// </summary>
public sealed record BitLinearConfig
{
    public BitLinearConfig(int inputDimension, int outputDimension)
    {
        if (inputDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputDimension), "Input dimension must be positive.");
        }

        if (outputDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputDimension), "Output dimension must be positive.");
        }

        InputDimension = inputDimension;
        OutputDimension = outputDimension;
    }

    public int InputDimension { get; }

    public int OutputDimension { get; }

    public bool Bias => false;
}
