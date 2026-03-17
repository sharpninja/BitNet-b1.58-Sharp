namespace BitNetSharp.Core;

public static class BitNetBootstrap
{
    public static (BitNetModel Model, TrainingReport Report) CreateSeededModel(VerbosityLevel verbosity = VerbosityLevel.Normal)
    {
        var model = BitNetModel.CreateDefault(verbosity);
        var report = new BitNetTrainer(model).TrainDefaults();
        return (model, report);
    }
}
