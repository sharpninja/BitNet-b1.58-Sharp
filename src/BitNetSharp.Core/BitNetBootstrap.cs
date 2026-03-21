namespace BitNetSharp.Core;

public static class BitNetBootstrap
{
    public static BitNetPaperModel CreatePaperModel(
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false) =>
        BitNetPaperModel.CreateDefault(verbosity, enableChainBuckets, enableSequenceCompression);

    public static BitNetPaperModel CreatePaperModel(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false) =>
        BitNetPaperModel.CreateForTrainingCorpus(trainingExamples, verbosity, enableChainBuckets, enableSequenceCompression);
}
