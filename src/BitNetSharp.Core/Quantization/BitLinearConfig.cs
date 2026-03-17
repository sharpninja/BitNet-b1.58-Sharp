namespace BitNetSharp.Core.Quantization;

public sealed record BitLinearConfig
{
    public BitLinearConfig(int inputDimension, int outputDimension, bool bias = false)
    {
        if (inputDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(inputDimension), "Input dimension must be positive.");
        }

        if (outputDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(outputDimension), "Output dimension must be positive.");
        }

        if (bias)
        {
            throw new ArgumentException("BitLinear does not support bias parameters.", nameof(bias));
        }

        InputDimension = inputDimension;
        OutputDimension = outputDimension;
        Bias = bias;
    }

    public int InputDimension { get; }

    public int OutputDimension { get; }

    public bool Bias { get; }
}
