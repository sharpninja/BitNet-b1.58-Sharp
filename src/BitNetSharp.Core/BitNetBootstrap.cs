using Microsoft.Extensions.Logging;

namespace BitNetSharp.Core;

public static class BitNetBootstrap
{
    public static BitNetPaperModel CreatePaperModel(
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false,
        ILoggerFactory? loggerFactory = null) =>
        BitNetPaperModel.CreateDefault(verbosity, enableChainBuckets, enableSequenceCompression, loggerFactory);

    public static BitNetPaperModel CreatePaperModel(
        IEnumerable<TrainingExample> trainingExamples,
        VerbosityLevel verbosity = VerbosityLevel.Normal,
        bool enableChainBuckets = false,
        bool enableSequenceCompression = false,
        ILoggerFactory? loggerFactory = null) =>
        BitNetPaperModel.CreateForTrainingCorpus(trainingExamples, verbosity, enableChainBuckets, enableSequenceCompression, loggerFactory);
}
