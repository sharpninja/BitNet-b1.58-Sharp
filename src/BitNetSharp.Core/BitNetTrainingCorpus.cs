namespace BitNetSharp.Core;

public static class BitNetTrainingCorpus
{
    public const string BenchmarkDatasetName = "TinyLlama-1.1B";

    public static IReadOnlyList<TrainingExample> CreateDefaultExamples() =>
    [
        new("hello", "Hello! I am BitNet Sharp, your compact American English assistant."),
        new("what are you", "I am a BitNet b1.58 inspired language model written in C# for .NET 10."),
        new("how do I train this model", "Use the training command to fit ternary weights with your examples and review the loss chart."),
        new("show visualization", "Use the visualize command to inspect the loss curve and the ternary weight histogram."),
        new("what language do you use", "I default to American English so the interaction style stays clear and consistent."),
        new("how are you hosted", "I prioritize Microsoft Agent Framework hosting with a local BitNet chat client.")
    ];

    public static IReadOnlyList<TrainingExample> CreateBenchmarkExamples() =>
    [
        new("which model anchors this benchmark", "TinyLlama 1 1B anchors the shared benchmark training slice for both local models."),
        new("how do I compare perplexity", "Use the benchmark report to compare WikiText2 perplexity after TinyLlama 1 1B training."),
        new("what does the paper model train on", "The paper aligned BitNet model fine tunes ternary output weights on the TinyLlama 1 1B benchmark slice."),
        new("what does the traditional model train on", "The traditional local model optimizes tensor softmax logits on the same TinyLlama 1 1B slice."),
        new("how are you hosted", "Both benchmark models stay in process with Microsoft Agent Framework hosting and local diagnostics."),
        new("what language do you use", "Benchmark prompts and diagnostics stay in clear American English.")
    ];

    public static IReadOnlyList<string> CreateDefaultVocabulary() =>
        CreateVocabulary(
            CreateDefaultExamples(),
            ["american", "english", "agent", "framework", "training", "visualize", "weights", "chart", "hosted"]);

    public static IReadOnlyList<string> CreateBenchmarkVocabulary() =>
        CreateVocabulary(
            CreateBenchmarkExamples(),
            ["tinyllama", "wikitext2", "perplexity", "benchmark", "american", "english", "agent", "framework", "hosting", "tensor", "ternary"]);

    public static IReadOnlyList<string> CreateVocabulary(IEnumerable<TrainingExample> examples, IEnumerable<string>? additionalTokens = null)
    {
        ArgumentNullException.ThrowIfNull(examples);

        return examples
            .SelectMany(example => $"{example.Prompt} {example.Response}".Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            .Select(token => token.Trim().ToLowerInvariant().Trim(',', '.', '!', '?', ';', ':'))
            .Where(token => !string.IsNullOrWhiteSpace(token))
            .Concat(additionalTokens ?? [])
            .Distinct(StringComparer.Ordinal)
            .ToArray();
    }
}
