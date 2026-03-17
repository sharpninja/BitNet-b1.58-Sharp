namespace BitNetSharp.Core;

public static class BitNetTrainingCorpus
{
    public static IReadOnlyList<TrainingExample> CreateDefaultExamples() =>
    [
        new("hello", "Hello! I am BitNet Sharp, your compact American English assistant."),
        new("what are you", "I am a BitNet b1.58 inspired language model written in C# for .NET 10."),
        new("how do I train this model", "Use the training command to fit ternary weights with your examples and review the loss chart."),
        new("show visualization", "Use the visualize command to inspect the loss curve and the ternary weight histogram."),
        new("what language do you use", "I default to American English so the interaction style stays clear and consistent."),
        new("how are you hosted", "I prioritize Microsoft Agent Framework hosting with a local BitNet chat client.")
    ];

    public static IReadOnlyList<string> CreateDefaultVocabulary() =>
        CreateDefaultExamples()
            .SelectMany(example => $"{example.Prompt} {example.Response}".Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .Select(token => token.Trim().ToLowerInvariant().Trim(',', '.', '!', '?', ';', ':'))
            .Where(token => !string.IsNullOrWhiteSpace(token))
            .Concat(["american", "english", "agent", "framework", "training", "visualize", "weights", "chart", "hosted"])
            .Distinct(StringComparer.Ordinal)
            .ToArray();
}
