using BitNetSharp.App;
using BitNetSharp.Core;
using System.Text.Json;

namespace BitNetSharp.Tests;

public sealed class DataGenTests
{
    [Fact]
    public void ParseDataGenCommandOptionsSupportsExtendedPromptOptions()
    {
        var options = DataGenCommandOptions.Parse(
            [
                "datagen",
                "--domain=code-review",
                "--count=2",
                "--output=data/code-review.jsonl",
                "--task-type=qa",
                "--constraint=Use American English",
                "--constraints=Grounded,Diverse",
                "--seeds=/tmp/seeds.json",
                "--output-schema={\"instruction\":\"string\",\"response\":\"string\"}",
                "--template=/tmp/template.json",
                "--lora=/tmp/domain-lora.bin",
                "--candidate-count=5",
                "--min-quality=0.7",
                "--max-tokens=64"
            ]);

        Assert.Equal("code-review", options.Domain);
        Assert.Equal(2, options.Count);
        Assert.EndsWith(Path.Combine("data", "code-review.jsonl"), options.OutputPath, StringComparison.Ordinal);
        Assert.Equal("qa", options.TaskType);
        Assert.Equal(["Use American English", "Grounded", "Diverse"], options.Constraints);
        Assert.Equal("/tmp/seeds.json", options.SeedsPath);
        Assert.Contains("\"instruction\"", options.OutputSchema, StringComparison.Ordinal);
        Assert.Equal("/tmp/template.json", options.TemplatePath);
        Assert.Equal("/tmp/domain-lora.bin", options.LoraPath);
        Assert.Equal(5, options.CandidateCount);
        Assert.Equal(0.7d, options.MinimumQualityScore);
        Assert.Equal(64, options.MaxOutputTokens);
    }

    [Fact]
    public async Task DataGenCommandWritesMergedPromptAndMetadata()
    {
        var seedPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}-seeds.json");
        var templatePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}-template.json");
        var outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.jsonl");

        File.WriteAllText(
            seedPath,
            """
            [
              { "instruction": "Review null handling", "response": "Check nullable flow and guard clauses." },
              { "prompt": "Review tests", "answer": "Prefer focused regression coverage." }
            ]
            """);

        File.WriteAllText(
            templatePath,
            """
            {
              "name": "test-template",
              "systemPrompt": "You are DataGen for {domain}.",
              "userPrompt": "Variation {variation}. Seed: {seed_instruction}. Baseline: {seed_response}. Constraints: {constraints}. Seeds: {seed_examples}. Schema: {output_schema}",
              "defaultOutputSchema": "{\"instruction\":\"string\",\"response\":\"string\"}"
            }
            """);

        try
        {
            var savedPath = await DataGenCommand.RunAsync(
                [
                    "datagen",
                    "--domain=code-review",
                    "--count=2",
                    $"--output={outputPath}",
                    $"--seeds={seedPath}",
                    $"--template={templatePath}",
                    "--constraint=Use American English",
                    "--candidate-count=2",
                    "--min-quality=0.5",
                    "--lora=/tmp/code-review-lora.bin"
                ],
                VerbosityLevel.Quiet);

            Assert.Equal(outputPath, savedPath);

            var lines = File.ReadAllLines(outputPath);
            Assert.Equal(2, lines.Length);

            var first = JsonSerializer.Deserialize<DataGenDatasetEntry>(lines[0]);
            Assert.NotNull(first);
            Assert.Equal("code-review", first.Domain);
            Assert.Equal("instruction-response", first.TaskType);
            Assert.True(first.QualityScore >= 0.5d);
            Assert.NotEmpty(first.Prompt);
            Assert.Contains("Variation pattern-", first.Prompt, StringComparison.Ordinal);
            Assert.Contains("Review", first.Prompt, StringComparison.Ordinal);
            Assert.NotEmpty(first.GroundingContext);
            Assert.Equal("/tmp/code-review-lora.bin", first.LoraPath);
            Assert.False(string.IsNullOrWhiteSpace(first.SeedInstruction));
            Assert.False(string.IsNullOrWhiteSpace(first.SeedResponse));
            Assert.False(string.IsNullOrWhiteSpace(first.Variation));
            Assert.Equal("bitnet-b1.58-sharp", first.GeneratorModel);
            Assert.Contains("synthetic", first.Tags);
        }
        finally
        {
            File.Delete(seedPath);
            File.Delete(templatePath);
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }

    [Fact]
    public async Task DataGenCommandCanGenerateWithoutExplicitSeeds()
    {
        var outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.jsonl");

        try
        {
            await DataGenCommand.RunAsync(
                [
                    "datagen",
                    "--domain=education",
                    "--count=1",
                    $"--output={outputPath}",
                    "--task-type=classification",
                    "--constraint=Use American English"
                ],
                VerbosityLevel.Quiet);

            var line = File.ReadAllText(outputPath);
            var entry = JsonSerializer.Deserialize<DataGenDatasetEntry>(line);

            Assert.NotNull(entry);
            Assert.Equal("education", entry.Domain);
            Assert.Equal("classification", entry.TaskType);
            Assert.Single(entry.GroundingContext);
            Assert.Contains("education", entry.SeedInstruction, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }
}
