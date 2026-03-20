using BitNetSharp.App;
using BitNetSharp.Core;
using System.Text.Json;

namespace BitNetSharp.Tests;

public sealed class DataGenTests
{
    [Fact]
    public void ParseDataGenOptionsSupportsDomainSeedsConstraintsAndLora()
    {
        var options = DataGenOptions.Parse(
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
                "--lora=/tmp/domain-lora.bin",
                "--candidate-count=5",
                "--min-quality=0.7",
                "--max-tokens=64"
            ],
            HostedAgentModelFactory.DefaultModelId,
            VerbosityLevel.Verbose);

        Assert.Equal("code-review", options.Domain);
        Assert.Equal(2, options.Count);
        Assert.EndsWith(Path.Combine("data", "code-review.jsonl"), options.OutputPath, StringComparison.Ordinal);
        Assert.Equal("qa", options.TaskType);
        Assert.Equal(["Use American English", "Grounded", "Diverse"], options.Constraints);
        Assert.Equal("/tmp/seeds.json", options.SeedPath);
        Assert.Contains("\"instruction\"", options.OutputSchema, StringComparison.Ordinal);
        Assert.Equal("/tmp/domain-lora.bin", options.LoraPath);
        Assert.Equal(5, options.CandidateCount);
        Assert.Equal(0.7d, options.MinimumQualityScore);
        Assert.Equal(64, options.MaxOutputTokens);
    }

    [Fact]
    public void LoadSeedsAcceptsInstructionAndPromptAliases()
    {
        var seedPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.json");
        File.WriteAllText(
            seedPath,
            """
            [
              { "instruction": "Explain a bug", "response": "Summarize the failing path." },
              { "prompt": "Review a patch", "response": "Focus on correctness and tests." }
            ]
            """);

        try
        {
            var seeds = DataGenSeedExample.LoadMany(seedPath);

            Assert.Collection(
                seeds,
                seed => Assert.Equal("Explain a bug", seed.Instruction),
                seed => Assert.Equal("Review a patch", seed.Instruction));
        }
        finally
        {
            File.Delete(seedPath);
        }
    }

    [Fact]
    public async Task GeneratorProducesTrainingReadyJsonlDataset()
    {
        var seedPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}-seeds.json");
        var outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.jsonl");
        File.WriteAllText(
            seedPath,
            """
            [
              { "instruction": "Review null handling", "response": "Check nullable flow and guard clauses." },
              { "instruction": "Review tests", "response": "Prefer focused regression coverage." }
            ]
            """);

        try
        {
            var template = new DataGenPromptTemplate(
                "test-template",
                "You are DataGen for {domain}.",
                "Create sample {sample_number} of {count} using {seed_examples} and {constraints}. Schema: {output_schema}",
                DataGenOptions.DefaultOutputSchema);

            using var model = new StubHostedAgentModel(
                [
                    "Candidate alpha",
                    "Candidate alpha",
                    "Candidate beta",
                    "Candidate gamma",
                    "Candidate gamma",
                    "Candidate delta"
                ]);

            var generator = new DataGenGenerator(model, template);
            var options = new DataGenOptions(
                "code-review",
                2,
                outputPath,
                "instruction-response",
                ["Use American English"],
                seedPath,
                DataGenOptions.DefaultOutputSchema,
                null,
                "/tmp/code-review-lora.bin",
                HostedAgentModelFactory.DefaultModelId,
                VerbosityLevel.Normal,
                CandidateCount: 3,
                MinimumQualityScore: 0.45d,
                MaxOutputTokens: 32);

            var dataset = await generator.GenerateAsync(options);
            DataGenGenerator.WriteJsonl(outputPath, dataset);

            var lines = File.ReadAllLines(outputPath);
            Assert.Equal(2, lines.Length);

            var first = JsonSerializer.Deserialize<DataGenDatasetEntry>(lines[0]);
            Assert.NotNull(first);
            Assert.Equal("code-review", first.Domain);
            Assert.Equal("instruction-response", first.TaskType);
            Assert.True(first.QualityScore >= 0.45d);
            Assert.Contains("code-review", first.Response, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("instruction-response", first.Response, StringComparison.OrdinalIgnoreCase);
            Assert.NotEmpty(first.GroundingContext);
            Assert.Equal("/tmp/code-review-lora.bin", first.LoraPath);
        }
        finally
        {
            File.Delete(seedPath);
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }

    private sealed class StubHostedAgentModel(IReadOnlyList<string> responses) : IHostedAgentModel
    {
        private int _responseIndex;

        public string AgentName => "stub-datagen";

        public string ModelId => "stub-datagen";

        public string DisplayName => "Stub DataGen";

        public string PrimaryLanguage => "en-US";

        public VerbosityLevel Verbosity => VerbosityLevel.Normal;

        public string SystemPrompt => "Stub system prompt";

        public IReadOnlyList<string> DescribeModel() => ["Stub DataGen"];

        public void Dispose()
        {
        }

        public Task<HostedAgentModelResponse> GetResponseAsync(
            string prompt,
            int? maxOutputTokens = null,
            CancellationToken cancellationToken = default)
        {
            var response = responses[_responseIndex % responses.Count];
            _responseIndex++;
            return Task.FromResult(new HostedAgentModelResponse(response, [$"Prompt length: {prompt.Length}"]));
        }
    }
}
