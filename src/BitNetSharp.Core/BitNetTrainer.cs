namespace BitNetSharp.Core;

public sealed class BitNetTrainer
{
    private readonly BitNetModel _model;

    public BitNetTrainer(BitNetModel model)
    {
        _model = model;
    }

    public TrainingReport TrainDefaults(int epochs = 3) =>
        _model.Train(BitNetTrainingCorpus.CreateDefaultExamples(), epochs);

    public TrainingReport Train(IEnumerable<TrainingExample> examples, int epochs = 3) =>
        _model.Train(examples, epochs);
}
