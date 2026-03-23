namespace BitNetSharp.Core.Training;

public sealed class AdamWOptimizer
{
    public AdamWOptimizer(
        float learningRate = 0.05f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        float weightDecay = 0.01f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);
        if (beta1 <= 0f || beta1 >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(beta1), "Beta1 must be between 0 and 1.");
        }

        if (beta2 <= 0f || beta2 >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(beta2), "Beta2 must be between 0 and 1.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        ArgumentOutOfRangeException.ThrowIfNegative(weightDecay);

        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
    }

    public float LearningRate { get; }

    public float Beta1 { get; }

    public float Beta2 { get; }

    public float Epsilon { get; }

    public float WeightDecay { get; }

    public OptimizerState CreateState(int rows, int columns) => new(rows, columns);

    public void Step(float[,] parameters, float[,] gradients, OptimizerState state)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(gradients);
        ArgumentNullException.ThrowIfNull(state);

        if (parameters.GetLength(0) != gradients.GetLength(0) || parameters.GetLength(1) != gradients.GetLength(1))
        {
            throw new ArgumentException("Parameter and gradient shapes must match.");
        }

        if (state.FirstMoment.GetLength(0) != parameters.GetLength(0) || state.FirstMoment.GetLength(1) != parameters.GetLength(1))
        {
            throw new ArgumentException("Optimizer state shape must match the parameter shape.", nameof(state));
        }

        state.Step++;

        var biasCorrection1 = 1f - MathF.Pow(Beta1, state.Step);
        var biasCorrection2 = 1f - MathF.Pow(Beta2, state.Step);

        for (var row = 0; row < parameters.GetLength(0); row++)
        {
            for (var column = 0; column < parameters.GetLength(1); column++)
            {
                var gradient = gradients[row, column];
                var firstMoment = (Beta1 * state.FirstMoment[row, column]) + ((1f - Beta1) * gradient);
                var secondMoment = (Beta2 * state.SecondMoment[row, column]) + ((1f - Beta2) * gradient * gradient);
                state.FirstMoment[row, column] = firstMoment;
                state.SecondMoment[row, column] = secondMoment;

                var correctedFirstMoment = firstMoment / MathF.Max(biasCorrection1, Epsilon);
                var correctedSecondMoment = secondMoment / MathF.Max(biasCorrection2, Epsilon);
                var update = correctedFirstMoment / (MathF.Sqrt(correctedSecondMoment) + Epsilon);
                update += WeightDecay * parameters[row, column];
                parameters[row, column] -= LearningRate * update;
            }
        }
    }

    public sealed class OptimizerState
    {
        public OptimizerState(int rows, int columns)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);

            FirstMoment = new float[rows, columns];
            SecondMoment = new float[rows, columns];
        }

        public float[,] FirstMoment { get; }

        public float[,] SecondMoment { get; }

        public long Step { get; internal set; }
    }
}
