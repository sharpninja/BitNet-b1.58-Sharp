using System.Text.Json;
using BitNetSharp.Core;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Tests;

public sealed class BitNetDataLoaderTests
{
    [Fact]
    public void SmallCorpusFixturesNormalizeToExpectedPromptResponsePairs()
    {
        var train = LoadNormalizedPairs(SmallCorpusTrainPath).ToArray();
        var valid = LoadNormalizedPairs(SmallCorpusValidPath).ToArray();

        Assert.Equal(
            [
                new TrainingExample("alpha beta", "gamma delta"),
                new TrainingExample("pack fixed batches\n\nkeep sequence order", "preserve context"),
                new TrainingExample("preserve order", "teacher forcing stays intact")
            ],
            train);

        Assert.Equal(
            [
                new TrainingExample("what stays held out?", "validation tokens stay held out.")
            ],
            valid);

        Assert.Empty(train.Intersect(valid));
    }

    [Fact]
    public void SmallCorpusSourceSplitsDeterministicallyIntoPackedTrainingAndHeldOutValidationWindows()
    {
        var examples = LoadNormalizedPairs(SmallCorpusSourcePath).ToArray();
        var loader = CreateLoader(examples, sequenceLength: 4, batchSize: 2, validationFraction: 0.1d, testFraction: 0d, seed: 4);

        var splitSequences = loader.Load(examples);

        Assert.Equal(4, splitSequences[BitNetDataSplit.Training].Count);
        Assert.Equal(2, splitSequences[BitNetDataSplit.Validation].Count);
        Assert.Empty(splitSequences[BitNetDataSplit.Test]);

        AssertWindows(
            splitSequences[BitNetDataSplit.Training],
            BitNetDataSplit.Training,
            ("training-window-0", 5),
            ("training-window-1", 5),
            ("training-window-2", 5),
            ("training-window-3", 5));

        AssertWindows(
            splitSequences[BitNetDataSplit.Validation],
            BitNetDataSplit.Validation,
            ("validation-window-0", 5),
            ("validation-window-1", 5));
    }

    [Fact]
    public void SmallCorpusSourceCreatesDeterministicBatchesFromPackedTrainingAndValidationWindows()
    {
        var examples = LoadNormalizedPairs(SmallCorpusSourcePath).ToArray();
        var loader = CreateLoader(examples, sequenceLength: 4, batchSize: 2, validationFraction: 0.1d, testFraction: 0d, seed: 4);

        var trainingBatches = loader.CreateBatches(examples, BitNetDataSplit.Training);
        Assert.Equal(2, trainingBatches.Count);
        Assert.Collection(
            trainingBatches,
            batch =>
            {
                Assert.Equal(BitNetDataSplit.Training, batch.Split);
                Assert.Equal(0, batch.BatchIndex);
                Assert.Equal(0, batch.EpochIndex);
                Assert.Equal(2, batch.SequenceCount);
                Assert.Equal(10, batch.TokenCount);
                AssertBatchWindows(
                    batch,
                    ("training-window-0", 5),
                    ("training-window-1", 5));
            },
            batch =>
            {
                Assert.Equal(BitNetDataSplit.Training, batch.Split);
                Assert.Equal(1, batch.BatchIndex);
                Assert.Equal(0, batch.EpochIndex);
                Assert.Equal(2, batch.SequenceCount);
                Assert.Equal(10, batch.TokenCount);
                AssertBatchWindows(
                    batch,
                    ("training-window-2", 5),
                    ("training-window-3", 5));
            });

        var validationBatches = loader.CreateBatches(examples, BitNetDataSplit.Validation);
        Assert.Single(validationBatches);
        Assert.Equal(0, validationBatches[0].BatchIndex);
        Assert.Equal(0, validationBatches[0].EpochIndex);
        Assert.Equal(BitNetDataSplit.Validation, validationBatches[0].Split);
        Assert.Equal(2, validationBatches[0].SequenceCount);
        Assert.Equal(10, validationBatches[0].TokenCount);
        AssertBatchWindows(
            validationBatches[0],
            ("validation-window-0", 5),
            ("validation-window-1", 5));
    }

    private static string SmallCorpusSourcePath => FixturePath("small-corpus.source.jsonl");

    private static string SmallCorpusTrainPath => FixturePath("small-corpus.train.jsonl");

    private static string SmallCorpusValidPath => FixturePath("small-corpus.valid.jsonl");

    private static string FixturePath(string fileName) =>
        Path.GetFullPath(
            Path.Combine(
                AppContext.BaseDirectory,
                "..",
                "..",
                "..",
                "Fixtures",
                "Training",
                fileName));

    private static BitNetDataLoader CreateLoader(
        IReadOnlyList<TrainingExample> examples,
        int sequenceLength,
        int batchSize,
        double validationFraction,
        double testFraction,
        int seed)
    {
        var vocabulary = BitNetTrainingCorpus.CreateVocabulary(examples);
        return new BitNetDataLoader(
            vocabulary,
            new BitNetDataLoaderOptions(
                sequenceLength: sequenceLength,
                batchSize: batchSize,
                validationFraction: validationFraction,
                testFraction: testFraction,
                shuffle: false,
                dropLast: true,
                seed: seed));
    }

    private static IEnumerable<TrainingExample> LoadNormalizedPairs(string path)
    {
        foreach (var record in LoadJsonLines(path))
        {
            yield return NormalizePair(record);
        }
    }

    private static TrainingExample NormalizePair(JsonElement record)
    {
        var prompt = GetPrompt(record);
        var response = GetResponse(record);
        return new TrainingExample(prompt, response);
    }

    private static string GetPrompt(JsonElement record)
    {
        var prompt = TryGetString(record, "prompt");
        if (!string.IsNullOrWhiteSpace(prompt))
        {
            return prompt;
        }

        var instruction = TryGetString(record, "instruction");
        if (!string.IsNullOrWhiteSpace(instruction))
        {
            var input = TryGetString(record, "input", "context");
            return string.IsNullOrWhiteSpace(input) ? instruction : $"{instruction}\n\n{input}";
        }

        if (TryGetArray(record, "messages", "conversations") is { } messages)
        {
            foreach (var message in messages)
            {
                var role = TryGetString(message, "role", "from", "speaker");
                var content = TryGetString(message, "content", "value", "text");
                if (string.IsNullOrWhiteSpace(role) || string.IsNullOrWhiteSpace(content))
                {
                    continue;
                }

                if (IsUserRole(role))
                {
                    return content;
                }
            }
        }

        throw new InvalidOperationException("The small corpus fixture record does not expose a prompt shape.");
    }

    private static string GetResponse(JsonElement record)
    {
        var response = TryGetString(record, "response", "output");
        if (!string.IsNullOrWhiteSpace(response))
        {
            return response;
        }

        if (TryGetArray(record, "messages", "conversations") is { } messages)
        {
            foreach (var message in messages)
            {
                var role = TryGetString(message, "role", "from", "speaker");
                var content = TryGetString(message, "content", "value", "text");
                if (string.IsNullOrWhiteSpace(role) || string.IsNullOrWhiteSpace(content))
                {
                    continue;
                }

                if (IsAssistantRole(role))
                {
                    return content;
                }
            }
        }

        throw new InvalidOperationException("The small corpus fixture record does not expose a response shape.");
    }

    private static bool IsUserRole(string role) =>
        role.Equals("user", StringComparison.OrdinalIgnoreCase)
        || role.Equals("human", StringComparison.OrdinalIgnoreCase)
        || role.Equals("prompt", StringComparison.OrdinalIgnoreCase)
        || role.Equals("instruction", StringComparison.OrdinalIgnoreCase);

    private static bool IsAssistantRole(string role) =>
        role.Equals("assistant", StringComparison.OrdinalIgnoreCase)
        || role.Equals("gpt", StringComparison.OrdinalIgnoreCase)
        || role.Equals("model", StringComparison.OrdinalIgnoreCase)
        || role.Equals("response", StringComparison.OrdinalIgnoreCase)
        || role.Equals("output", StringComparison.OrdinalIgnoreCase);

    private static void AssertWindows(
        IReadOnlyList<BitNetTokenSequence> sequences,
        BitNetDataSplit split,
        params (string Source, int Length)[] expected)
    {
        Assert.Equal(expected.Length, sequences.Count);

        for (var index = 0; index < expected.Length; index++)
        {
            var (source, length) = expected[index];
            Assert.Equal(split, sequences[index].Split);
            Assert.Equal(source, sequences[index].Source);
            Assert.Equal(length, sequences[index].Length);
        }
    }

    private static void AssertBatchWindows(
        TrainingBatch batch,
        params (string Source, int Length)[] expected)
    {
        Assert.Equal(expected.Length, batch.Sequences.Count);

        for (var index = 0; index < expected.Length; index++)
        {
            var (source, length) = expected[index];
            Assert.Equal(source, batch.Sequences[index].Source);
            Assert.Equal(length, batch.Sequences[index].Length);
        }
    }

    private static IEnumerable<JsonElement> LoadJsonLines(string path)
    {
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var document = JsonDocument.Parse(line);
            yield return document.RootElement.Clone();
        }
    }

    private static JsonElement[]? TryGetArray(JsonElement record, params string[] names)
    {
        foreach (var name in names)
        {
            if (record.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Array)
            {
                return value.EnumerateArray().Select(static item => item.Clone()).ToArray();
            }
        }

        return null;
    }

    private static string? TryGetString(JsonElement record, params string[] names)
    {
        foreach (var name in names)
        {
            if (record.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String)
            {
                var text = value.GetString();
                if (!string.IsNullOrWhiteSpace(text))
                {
                    return text;
                }
            }
        }

        return null;
    }
}
