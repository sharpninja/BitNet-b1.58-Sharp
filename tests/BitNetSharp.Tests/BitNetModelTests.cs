using BitNetSharp.App;
using BitNetSharp.Core;
using BitNetSharp.Core.Training;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.DependencyInjection;
using System.Reflection;
using System.Text.Json;

namespace BitNetSharp.Tests;

public sealed class BitNetPaperModelTests
{
    [Fact]
    public void GeneratedResponseUsesPaperAlignedTransformerDiagnostics()
    {
        var model = BitNetBootstrap.CreatePaperModel(VerbosityLevel.Normal);
        var result = model.GenerateResponse("how are you hosted");

        Assert.False(string.IsNullOrWhiteSpace(result.ResponseText));
        Assert.DoesNotContain("Top next-token predictions:", result.ResponseText, StringComparison.Ordinal);
        Assert.NotEmpty(result.Tokens);
        Assert.Contains("microsoft agent framework", result.ResponseText, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("decoder-only transformer", result.Diagnostics[1], StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GeneratedResponseUsesTernaryPredictionsForUnmemorizedPrompt()
    {
        var model = new BitNetModel(new BitNetOptions(["alpha", "beta", "gamma"], VerbosityLevel.Quiet));
        model.Train([new TrainingExample("alpha", "beta gamma")], epochs: 1);

        var result = model.GenerateResponse("beta");

        Assert.Equal("gamma", result.ResponseText);
        Assert.Empty(result.Diagnostics);
    }

    [Fact]
    public void QuantizePreservesLargeCountsBeyondIntRange()
    {
        var model = new BitNetModel(new BitNetOptions(["alpha", "beta", "gamma"], VerbosityLevel.Quiet));
        var getId = typeof(BitNetModel).GetMethod("GetId", BindingFlags.Instance | BindingFlags.NonPublic);
        var quantize = typeof(BitNetModel).GetMethod("Quantize", BindingFlags.Instance | BindingFlags.NonPublic);
        var weightsField = typeof(BitNetModel).GetField("_weights", BindingFlags.Instance | BindingFlags.NonPublic);
        var priorsField = typeof(BitNetModel).GetField("_priors", BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(getId);
        Assert.NotNull(quantize);
        Assert.NotNull(weightsField);
        Assert.NotNull(priorsField);

        var alphaId = (int)getId.Invoke(model, ["alpha"])!;
        var betaId = (int)getId.Invoke(model, ["beta"])!;
        var gammaId = (int)getId.Invoke(model, ["gamma"])!;
        var weights = (sbyte[,])weightsField.GetValue(model)!;
        var vocabularySize = weights.GetLength(0);
        var counts = new long[vocabularySize, vocabularySize];
        var priors = new long[vocabularySize];
        var baseCount = (long)int.MaxValue + 100L;

        for (var column = 0; column < vocabularySize; column++)
        {
            counts[alphaId, column] = baseCount;
            priors[column] = baseCount;
        }

        counts[alphaId, betaId] = baseCount + (vocabularySize * 2L);
        priors[betaId] = baseCount + (vocabularySize * 2L);

        quantize.Invoke(model, [counts, priors]);

        weights = (sbyte[,])weightsField.GetValue(model)!;
        var scoredPriors = (float[])priorsField.GetValue(model)!;

        Assert.Equal(1, weights[alphaId, betaId]);
        Assert.Equal(-1, weights[alphaId, gammaId]);
        Assert.Equal(0.35f, scoredPriors[betaId]);
        Assert.Equal(0f, scoredPriors[gammaId]);
    }

    [Fact]
    public void VisualizationIncludesChartsAndCsv()
    {
        var model = BitNetBootstrap.CreatePaperModel();
        var stats = model.GetTernaryWeightStats();

        Assert.True(stats.NegativeCount > 0);
        Assert.True(stats.PositiveCount > 0);
        Assert.Equal(stats.TotalCount, stats.NegativeCount + stats.ZeroCount + stats.PositiveCount);
    }

    [Fact]
    public void AgentHostBuildsWithMicrosoftAgentFrameworkRegistration()
    {
        var model = BitNetBootstrap.CreatePaperModel();
        using var host = BitNetAgentHost.Build(model);
        var summary = host.Services.GetRequiredService<BitNetHostSummary>();

        Assert.Equal("bitnet-b1.58-sharp", summary.AgentName);
        Assert.Equal("bitnet-b1.58-sharp", summary.ModelId);
        Assert.Equal("Microsoft Agent Framework", summary.HostingFramework);
        Assert.Equal("en-US", summary.PrimaryLanguage);
    }

    [Fact]
    public async Task HostedAgentFactorySupportsTraditionalComparisonModel()
    {
        using var model = HostedAgentModelFactory.Create(HostedAgentModelFactory.TraditionalLocalModelId, VerbosityLevel.Normal);
        var response = await model.GetResponseAsync("how are you hosted");

        Assert.Equal(HostedAgentModelFactory.TraditionalLocalModelId, model.ModelId);
        Assert.False(string.IsNullOrWhiteSpace(response.Text));
        Assert.Contains("microsoft agent framework", response.Text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(response.Diagnostics, diagnostic => diagnostic.Contains("tensor-based ordered-context", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TraditionalLocalModelLearnsSimplePromptResponse()
    {
        var model = new TraditionalLocalModel(
            new BitNetOptions(["alpha", "beta", "gamma", "delta"], VerbosityLevel.Normal, MaxResponseTokens: 4),
            embeddingDimension: 16,
            contextWindow: 4,
            seed: 11);

        model.Train(
            [
                new TrainingExample("alpha beta", "gamma delta")
            ],
            epochs: 80,
            learningRate: 0.55f);

        var result = model.GenerateResponse("alpha beta", maxTokens: 2);

        Assert.Equal(["gamma", "delta"], result.Tokens);
        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Contains("tensor-based ordered-context", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public void TraditionalLocalTrainingReportIncludesWeightSignDistribution()
    {
        var model = new TraditionalLocalModel(
            new BitNetOptions(["alpha", "beta", "gamma", "delta"], VerbosityLevel.Quiet),
            embeddingDimension: 8,
            contextWindow: 4,
            seed: 19);

        var report = model.Train(
            [
                new TrainingExample("alpha beta", "gamma delta")
            ],
            epochs: 12,
            learningRate: 0.3f);

        Assert.True(report.NegativeWeights > 0);
        Assert.True(report.PositiveWeights > 0);
        Assert.Equal(report.NegativeWeights + report.ZeroWeights + report.PositiveWeights, model.GetTernaryWeightStats().TotalCount);
    }

    [Fact]
    public void TraditionalHostedAgentModelExposesInspectableWeightStats()
    {
        using var model = HostedAgentModelFactory.Create(HostedAgentModelFactory.TraditionalLocalModelId, VerbosityLevel.Quiet);

        var inspectable = Assert.IsAssignableFrom<IInspectableHostedAgentModel>(model);
        var stats = inspectable.GetTernaryWeightStats();

        Assert.True(stats.TotalCount > 0);
        Assert.True(stats.NegativeCount > 0);
        Assert.True(stats.PositiveCount > 0);
    }

    [Fact]
    public void PaperModelTrainingLearnsTeacherForcedContinuationBeyondFirstResponseToken()
    {
        var model = new BitNetPaperModel(
            new BitNetOptions(["alpha", "beta", "gamma", "delta"], VerbosityLevel.Quiet, MaxResponseTokens: 2),
            new BitNetConfig(vocabSize: 7, dimension: 16, hiddenDimension: 32, layerCount: 2, headCount: 4, maxSequenceLength: 16),
            seed: 17);
        var exportScale = typeof(BitNetPaperModel)
            .GetMethod("ExportFinalNormScale", BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(exportScale);

        var baselineScale = (float[])exportScale.Invoke(model, [])!;

        var report = model.Train(
            [
                new TrainingExample("alpha beta", "gamma delta")
            ],
            new BitNetTrainingOptions(
                epochs: 120,
                learningRate: 0.2f,
                weightDecay: 0f,
                evaluationInterval: 0,
                dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 16)));

        var result = model.GenerateResponse("alpha beta gamma", maxTokens: 1);
        var trainedScale = (float[])exportScale.Invoke(model, [])!;

        Assert.Equal(["delta"], result.Tokens);
        Assert.True(report.LossHistory[^1] < report.LossHistory[0]);
        Assert.Contains(
            trainedScale.Zip(baselineScale, static (after, before) => MathF.Abs(after - before)),
            static delta => delta > 1e-4f);
    }

    [Fact]
    public void PaperModelTrainingUpdatesFinalNormAndOutputHeadOnly()
    {
        var model = new BitNetPaperModel(
            new BitNetOptions(["alpha", "beta", "gamma", "delta"], VerbosityLevel.Quiet, MaxResponseTokens: 2),
            new BitNetConfig(vocabSize: 7, dimension: 16, hiddenDimension: 32, layerCount: 2, headCount: 4, maxSequenceLength: 16),
            seed: 23);
        var exportNormScales = typeof(BitNetPaperModel).GetMethod("ExportNormScales", BindingFlags.Instance | BindingFlags.NonPublic);
        var exportTransformerProjectionWeights = typeof(BitNetPaperModel).GetMethod("ExportTransformerProjectionWeights", BindingFlags.Instance | BindingFlags.NonPublic);
        var exportOutputHeadWeights = typeof(BitNetPaperModel).GetMethod("ExportOutputHeadWeights", BindingFlags.Instance | BindingFlags.NonPublic);

        Assert.NotNull(exportNormScales);
        Assert.NotNull(exportTransformerProjectionWeights);
        Assert.NotNull(exportOutputHeadWeights);

        var beforeNormScales = ((IReadOnlyList<float[]>)exportNormScales.Invoke(model, [])!)
            .Select(static scale => scale.ToArray())
            .ToArray();
        var beforeProjectionWeights = ((IReadOnlyList<float[,]>)exportTransformerProjectionWeights.Invoke(model, [])!)
            .Select(CloneMatrix)
            .ToArray();
        var beforeOutputHeadWeights = CloneMatrix((float[,])exportOutputHeadWeights.Invoke(model, [])!);

        model.Train(
            [
                new TrainingExample("alpha beta", "gamma delta")
            ],
            new BitNetTrainingOptions(
                epochs: 120,
                learningRate: 0.2f,
                weightDecay: 0f,
                evaluationInterval: 0,
                dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: 16)));

        var afterNormScales = ((IReadOnlyList<float[]>)exportNormScales.Invoke(model, [])!)
            .Select(static scale => scale.ToArray())
            .ToArray();
        var afterProjectionWeights = ((IReadOnlyList<float[,]>)exportTransformerProjectionWeights.Invoke(model, [])!)
            .Select(CloneMatrix)
            .ToArray();
        var afterOutputHeadWeights = CloneMatrix((float[,])exportOutputHeadWeights.Invoke(model, [])!);

        Assert.True(VectorChanged(beforeNormScales[^1], afterNormScales[^1]));
        Assert.All(
            beforeNormScales
                .Take(beforeNormScales.Length - 1)
                .Zip(afterNormScales.Take(afterNormScales.Length - 1)),
            pair => Assert.False(VectorChanged(pair.First, pair.Second)));
        Assert.True(MatrixChanged(beforeOutputHeadWeights, afterOutputHeadWeights));
        Assert.All(
            beforeProjectionWeights.Zip(afterProjectionWeights),
            pair => Assert.False(MatrixChanged(pair.First, pair.Second)));
    }

    [Fact]
    public void BenchmarkOptionsIncludePrimaryAndComparisonModels()
    {
        var options = HostedAgentBenchmarkOptions.Parse(
            ["benchmark", "--model=bitnet-b1.58-sharp", "--compare-model=traditional-local", "--prompt=how are you hosted", "--max-tokens=3"],
            VerbosityLevel.Verbose);

        Assert.Equal(["bitnet-b1.58-sharp", "traditional-local"], options.ModelSpecifiers);
        Assert.Equal("how are you hosted", options.Prompt);
        Assert.Equal(3, options.MaxOutputTokens);
        Assert.Equal(VerbosityLevel.Verbose, options.Verbosity);
    }

    [Fact]
    public void DataGenSeedLoaderSupportsSeedAliases()
    {
        var seedPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}-datagen-seeds.json");
        File.WriteAllText(
            seedPath,
            """
            [
              {
                "prompt": "draft a medical triage question",
                "response": "summarize the complaint and first safety check"
              },
              {
                "instruction": "collect missing medication context",
                "answer": "review active prescriptions and contraindications"
              }
            ]
            """);

        try
        {
            var seeds = DataGenGenerator.LoadSeeds(seedPath);

            Assert.Collection(
                seeds,
                seed =>
                {
                    Assert.Equal("draft a medical triage question", seed.Instruction);
                    Assert.Equal("summarize the complaint and first safety check", seed.Response);
                },
                seed =>
                {
                    Assert.Equal("collect missing medication context", seed.Instruction);
                    Assert.Equal("review active prescriptions and contraindications", seed.Response);
                });
        }
        finally
        {
            File.Delete(seedPath);
        }
    }

    [Fact]
    public void DataGenGeneratorProducesStructuredExamples()
    {
        var generator = new DataGenGenerator(BitNetBootstrap.CreatePaperModel(VerbosityLevel.Quiet));
        var examples = generator.Generate(
                "medical-diagnosis",
                count: 3,
                seeds:
                [
                    new DataGenSeedExample(
                        "summarize the patient complaint",
                        "restate the complaint and note the top safety concern")
                ],
                loraAdapter: "/tmp/medical-lora.bin")
            .ToArray();

        Assert.Equal(3, examples.Length);
        Assert.All(
            examples,
            example =>
            {
                Assert.Equal("medical-diagnosis", example.Domain);
                Assert.Contains("medical-diagnosis", example.Instruction, StringComparison.Ordinal);
                Assert.Contains("medical-diagnosis", example.Response, StringComparison.Ordinal);
                Assert.Equal("bitnet-b1.58-sharp", example.GeneratorModel);
                Assert.Equal("medical-lora.bin", example.LoraAdapter);
                Assert.Contains("synthetic", example.Tags);
            });
        Assert.Contains(examples, example => example.Variation == "pattern-1");
        Assert.Contains(examples, example => example.Variation == "pattern-2");
        Assert.Contains(examples, example => example.Variation == "pattern-3");
    }

    [Fact]
    public void DataGenCommandOptionsParseSpaceSeparatedArguments()
    {
        var options = DataGenCommandOptions.Parse(
            [
                "datagen",
                "--domain", "medical-diagnosis",
                "--count", "12",
                "--seeds", "examples/seed-examples.json",
                "--output", "data/synthetic-medical.jsonl",
                "--lora", "medical-lora.bin"
            ]);

        Assert.Equal("medical-diagnosis", options.Domain);
        Assert.Equal(12, options.Count);
        Assert.EndsWith(Path.Combine("examples", "seed-examples.json"), options.SeedsPath, StringComparison.Ordinal);
        Assert.EndsWith(Path.Combine("data", "synthetic-medical.jsonl"), options.OutputPath, StringComparison.Ordinal);
        Assert.EndsWith("medical-lora.bin", options.LoraPath, StringComparison.Ordinal);
    }

    [Fact]
    public void DataGenCommandOptionsRejectEmptyRequiredValues()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => DataGenCommandOptions.Parse(
                [
                    "datagen",
                    "--domain=",
                    "--count", "12",
                    "--seeds", "examples/seed-examples.json",
                    "--output", "data/synthetic-medical.jsonl"
                ]));

        Assert.Contains("non-empty value", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DataGenCommandOptionsRejectOptionLikeNextToken()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => DataGenCommandOptions.Parse(
                [
                    "datagen",
                    "--domain", "--count",
                    "--count", "12",
                    "--seeds", "examples/seed-examples.json",
                    "--output", "data/synthetic-medical.jsonl"
                ]));

        Assert.Contains("requires a value", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DataGenCommandWritesJsonlDataset()
    {
        var outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}-synthetic.jsonl");
        var seedPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}-seeds.json");

        try
        {
            await File.WriteAllTextAsync(
                seedPath,
                """
                [
                  {
                    "prompt": "draft a support troubleshooting request",
                    "response": "summarize the issue and confirm the first diagnostic step"
                  }
                ]
                """);

            var writtenPath = await DataGenCommand.RunAsync(
                [
                    "datagen",
                    "--domain", "customer-support",
                    "--count", "2",
                    "--seeds", seedPath,
                    "--output", outputPath
                ],
                VerbosityLevel.Quiet);

            Assert.Equal(outputPath, writtenPath);

            var lines = await File.ReadAllLinesAsync(outputPath);
            Assert.Equal(2, lines.Length);

            using var document = JsonDocument.Parse(lines[0]);
            Assert.Equal("customer-support", document.RootElement.GetProperty("domain").GetString());
            Assert.Equal("bitnet-b1.58-sharp", document.RootElement.GetProperty("generatorModel").GetString());
            Assert.True(document.RootElement.GetProperty("instruction").GetString()!.Length > 0);
            Assert.True(document.RootElement.GetProperty("response").GetString()!.Length > 0);
        }
        finally
        {
            File.Delete(outputPath);
            File.Delete(seedPath);
        }
    }

    private static float[,] CloneMatrix(float[,] matrix)
    {
        var clone = new float[matrix.GetLength(0), matrix.GetLength(1)];
        Array.Copy(matrix, clone, matrix.Length);
        return clone;
    }

    private static bool MatrixChanged(float[,] before, float[,] after, float tolerance = 1e-4f)
    {
        Assert.Equal(before.GetLength(0), after.GetLength(0));
        Assert.Equal(before.GetLength(1), after.GetLength(1));

        for (var row = 0; row < before.GetLength(0); row++)
        {
            for (var column = 0; column < before.GetLength(1); column++)
            {
                if (MathF.Abs(before[row, column] - after[row, column]) > tolerance)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static bool VectorChanged(IReadOnlyList<float> before, IReadOnlyList<float> after, float tolerance = 1e-4f)
    {
        Assert.Equal(before.Count, after.Count);

        for (var index = 0; index < before.Count; index++)
        {
            if (MathF.Abs(before[index] - after[index]) > tolerance)
            {
                return true;
            }
        }

        return false;
    }
}
