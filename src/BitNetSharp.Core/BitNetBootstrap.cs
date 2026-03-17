namespace BitNetSharp.Core;

public static class BitNetBootstrap
{
    public static BitNetPaperModel CreatePaperModel(VerbosityLevel verbosity = VerbosityLevel.Normal) =>
        BitNetPaperModel.CreateDefault(verbosity);
}
