namespace BitNetSharp.Core;

public static class BitNetBootstrap
{
    public static BitNetPaperModel CreatePaperModel(VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        BitNetPaperModel.CreateDefault(verbosity);

    public static BitNetPaperModel CreatePaperModel(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        BitNetPaperModel.CreateForTrainingCorpus(trainingExamples, verbosity);
}
