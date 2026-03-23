namespace BitNetSharp.Core.Training;

public sealed record BitNetTrainingOptions
{
    public BitNetTrainingOptions(
        int epochs = 3,
        float learningRate = 0.05f,
        float weightDecay = 0.01f,
        float beta1 = 0.9f,
        float beta2 = 0.999f,
        float epsilon = 1e-8f,
        int evaluationInterval = 1,
        int checkpointInterval = 0,
        BitNetDataLoaderOptions? dataLoaderOptions = null,
        bool compactEvaluation = true,
        string? trainingDatasetName = null,
        string? validationDatasetName = null,
        string? checkpointDirectory = null,
        string? checkpointPrefix = null,
        Func<int, TrainingEvaluationSummary?>? externalEvaluation = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);
        ArgumentOutOfRangeException.ThrowIfNegative(weightDecay);
        if (beta1 <= 0f || beta1 >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(beta1), "Beta1 must be between 0 and 1.");
        }

        if (beta2 <= 0f || beta2 >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(beta2), "Beta2 must be between 0 and 1.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        ArgumentOutOfRangeException.ThrowIfNegative(evaluationInterval);
        ArgumentOutOfRangeException.ThrowIfNegative(checkpointInterval);
        if (checkpointPrefix is not null && string.IsNullOrWhiteSpace(checkpointPrefix))
        {
            throw new ArgumentException("Checkpoint prefix cannot be empty when provided.", nameof(checkpointPrefix));
        }

        Epochs = epochs;
        LearningRate = learningRate;
        WeightDecay = weightDecay;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        EvaluationInterval = evaluationInterval;
        CheckpointInterval = checkpointInterval;
        DataLoaderOptions = dataLoaderOptions ?? new BitNetDataLoaderOptions();
        CompactEvaluation = compactEvaluation;
        TrainingDatasetName = trainingDatasetName;
        ValidationDatasetName = validationDatasetName;
        CheckpointDirectory = string.IsNullOrWhiteSpace(checkpointDirectory)
            ? null
            : Path.GetFullPath(checkpointDirectory);
        CheckpointPrefix = string.IsNullOrWhiteSpace(checkpointPrefix)
            ? DefaultCheckpointPrefix
            : checkpointPrefix.Trim();
        ExternalEvaluation = externalEvaluation;
    }

    public const string DefaultCheckpointPrefix = "training-checkpoint";

    public int Epochs { get; }

    public float LearningRate { get; }

    public float WeightDecay { get; }

    public float Beta1 { get; }

    public float Beta2 { get; }

    public float Epsilon { get; }

    public int EvaluationInterval { get; }

    public int CheckpointInterval { get; }

    public BitNetDataLoaderOptions DataLoaderOptions { get; }

    public bool CompactEvaluation { get; }

    public string? TrainingDatasetName { get; }

    public string? ValidationDatasetName { get; }

    public string? CheckpointDirectory { get; }

    public string CheckpointPrefix { get; }

    public Func<int, TrainingEvaluationSummary?>? ExternalEvaluation { get; }
}
