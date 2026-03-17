namespace BitNetSharp.Core.Layers;

public abstract class Module
{
    public abstract float[,] Forward(float[,] input);

    public virtual float[,] BackwardSTE(float[,] gradientOutput)
    {
        ArgumentNullException.ThrowIfNull(gradientOutput);
        return (float[,])gradientOutput.Clone();
    }
}
