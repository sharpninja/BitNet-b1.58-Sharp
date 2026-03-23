using BitNetSharp.Core;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Core.Training;

public sealed class BitNetPaperTrainer
{
    private readonly BitNetPaperModel _model;
    private readonly BitNetTrainingOptions _options;
    private readonly BitNetDataLoader _loader;
    private readonly AdamWOptimizer _optimizer;
    private readonly AdamWOptimizer.OptimizerState _outputHeadOptimizerState;
    private readonly AdamWOptimizer.OptimizerState _finalNormOptimizerState;

    public BitNetPaperTrainer(BitNetPaperModel model, BitNetTrainingOptions? options = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _options = options ?? new BitNetTrainingOptions(dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: model.Config.MaxSequenceLength));
        _loader = new BitNetDataLoader(model.Options.Vocabulary, _options.DataLoaderOptions);
        _optimizer = new AdamWOptimizer(
            _options.LearningRate,
            _options.Beta1,
            _options.Beta2,
            _options.Epsilon,
            _options.WeightDecay);

        var weights = _model.ExportOutputHeadWeights();
        _outputHeadOptimizerState = _optimizer.CreateState(weights.GetLength(0), weights.GetLength(1));
        _finalNormOptimizerState = _optimizer.CreateState(1, _model.ExportFinalNormScale().Length);
    }

    public BitNetTrainingOptions Options => _options;

    public TrainingReport Train(IEnumerable<TrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        return Train(_loader, trainingSet);
    }

    public TrainingReport Train(BitNetDataLoader loader, IEnumerable<TrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(examples);

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        var splitSequences = loader.Load(trainingSet);
        var trainingSequences = splitSequences[BitNetDataSplit.Training];
        var validationSequences = splitSequences[BitNetDataSplit.Validation];
        if (trainingSequences.Count == 0)
        {
            var fallbackLoader = new BitNetDataLoader(
                _model.Options.Vocabulary,
                new BitNetDataLoaderOptions(
                    sequenceLength: _options.DataLoaderOptions.SequenceLength,
                    batchSize: _options.DataLoaderOptions.BatchSize,
                    validationFraction: 0d,
                    testFraction: 0d,
                    shuffle: _options.DataLoaderOptions.Shuffle,
                    dropLast: false,
                    seed: _options.DataLoaderOptions.Seed));

            splitSequences = fallbackLoader.Load(trainingSet);
            trainingSequences = splitSequences[BitNetDataSplit.Training];
            validationSequences = [];
        }

        if (trainingSequences.Count == 0)
        {
            throw new InvalidOperationException("The configured data split produced zero training sequences.");
        }

        var lossHistory = new List<double>(_options.Epochs);
        var epochMetrics = new List<TrainingEpochMetrics>(_options.Epochs);
        var evaluationSummaries = Array.Empty<TrainingEvaluationSummary>();
        var checkpoints = new List<TrainingCheckpointSummary>();
        var totalSamples = 0;

        for (var epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var batches = CreateBatches(trainingSequences, BitNetDataSplit.Training, epoch);
            var epochLoss = 0d;
            var epochObservations = 0;

            foreach (var batch in batches)
            {
                var batchResult = TrainBatch(batch);
                epochLoss += batchResult.TotalLoss;
                epochObservations += batchResult.Observations;
                totalSamples += batch.SequenceCount;
            }

            var averageLoss = epochObservations == 0 ? 0d : epochLoss / epochObservations;
            lossHistory.Add(averageLoss);

            var validationPerplexity = default(double?);
            if (_options.EvaluationInterval > 0 && (epoch + 1) % _options.EvaluationInterval == 0)
            {
                if (validationSequences.Count > 0)
                {
                    var validationMetrics = EvaluateSequences(validationSequences);
                    validationPerplexity = validationMetrics.Perplexity;
                    evaluationSummaries = evaluationSummaries
                        .Append(new TrainingEvaluationSummary(
                            _options.ValidationDatasetName ?? "HeldOut",
                            validationMetrics.Samples,
                            validationMetrics.AverageCrossEntropy,
                            validationMetrics.Perplexity))
                        .ToArray();
                    validationPerplexity = validationMetrics.Perplexity;
                }

                if (_options.ExternalEvaluation is not null)
                {
                    var externalSummary = _options.ExternalEvaluation(epoch + 1);
                    if (externalSummary is not null)
                    {
                        validationPerplexity ??= externalSummary.Perplexity;
                        evaluationSummaries = evaluationSummaries
                            .Append(externalSummary)
                            .ToArray();
                    }
                }

                var fixtureDatasets = BitNetBenchmarkFixtures.GetPerplexityDatasets(_options.CompactEvaluation);
                evaluationSummaries = evaluationSummaries
                    .Concat(EvaluateFixtures(fixtureDatasets))
                    .GroupBy(static summary => summary.Dataset, StringComparer.Ordinal)
                    .Select(static group => group.Last())
                    .ToArray();
            }

            epochMetrics.Add(new TrainingEpochMetrics(
                epoch + 1,
                averageLoss,
                totalSamples,
                epochObservations,
                validationPerplexity));

            if (_options.CheckpointInterval > 0 && (epoch + 1) % _options.CheckpointInterval == 0)
            {
                checkpoints.Add(SaveCheckpoint(epoch + 1, totalSamples));
            }
        }

        var stats = _model.GetTernaryWeightStats();
        return new TrainingReport(
            lossHistory,
            totalSamples,
            _options.Epochs,
            stats.NegativeCount,
            stats.ZeroCount,
            stats.PositiveCount,
            epochMetrics,
            evaluationSummaries,
            checkpoints,
            _options.TrainingDatasetName,
            validationSequences.Count > 0 ? _options.ValidationDatasetName ?? "HeldOut" : null);
    }

    private BatchResult TrainBatch(TrainingBatch batch)
    {
        var outputWeights = _model.ExportOutputHeadWeights();
        var outputGradients = new float[outputWeights.GetLength(0), outputWeights.GetLength(1)];
        var finalNormScale = _model.ExportFinalNormScale();
        var finalNormParameters = ToSingleRowMatrix(finalNormScale);
        var finalNormGradients = new float[1, finalNormScale.Length];
        var probabilities = new float[outputWeights.GetLength(0)];
        var totalLoss = 0d;
        var observations = 0;

        foreach (var sequence in batch.Sequences)
        {
            var inputIds = sequence.TokenIds.Take(sequence.TokenIds.Count - 1).ToArray();
            var targetIds = sequence.TokenIds.Skip(1).ToArray();
            var preHeadStates = _model.ForwardPreHeadStates(inputIds);

            for (var position = 0; position < targetIds.Length; position++)
            {
                var normalizedFeatures = GetRmsNormalizedRow(preHeadStates, position, _model.Config.RmsNormEpsilon);
                var features = ApplyScale(normalizedFeatures, finalNormScale);
                var loss = EvaluateStep(
                    outputWeights,
                    features,
                    normalizedFeatures,
                    targetIds[position],
                    probabilities,
                    outputGradients,
                    finalNormGradients);
                totalLoss += loss;
                observations++;
            }
        }

        _optimizer.Step(outputWeights, outputGradients, _outputHeadOptimizerState);
        _optimizer.Step(finalNormParameters, finalNormGradients, _finalNormOptimizerState);
        _model.ImportOutputHeadWeights(outputWeights);
        _model.ImportFinalNormScale(GetRow(finalNormParameters, 0));

        return new BatchResult(totalLoss, observations);
    }

    private static double EvaluateStep(
        float[,] weights,
        float[] features,
        float[] normalizedFeatures,
        int targetId,
        float[] probabilities,
        float[,] outputGradients,
        float[,] finalNormGradients)
    {
        var logits = new float[weights.GetLength(0)];
        for (var row = 0; row < weights.GetLength(0); row++)
        {
            var sum = 0f;
            for (var column = 0; column < weights.GetLength(1); column++)
            {
                sum += weights[row, column] * features[column];
            }

            logits[row] = sum;
        }

        var loss = CrossEntropyLoss.FromLogits(logits, targetId, probabilities);

        for (var row = 0; row < probabilities.Length; row++)
        {
            var gradient = probabilities[row] - (row == targetId ? 1f : 0f);
            for (var column = 0; column < features.Length; column++)
            {
                outputGradients[row, column] += gradient * features[column];
                finalNormGradients[0, column] += gradient * weights[row, column] * normalizedFeatures[column];
            }
        }

        return loss;
    }

    private static float[] GetRow(float[,] matrix, int rowIndex)
    {
        var result = new float[matrix.GetLength(1)];
        for (var column = 0; column < result.Length; column++)
        {
            result[column] = matrix[rowIndex, column];
        }

        return result;
    }

    private static float[] GetRmsNormalizedRow(float[,] matrix, int rowIndex, float epsilon)
    {
        var result = GetRow(matrix, rowIndex);
        var sumSquares = 0f;
        for (var column = 0; column < result.Length; column++)
        {
            sumSquares += result[column] * result[column];
        }

        var rms = MathF.Sqrt(sumSquares / result.Length + epsilon);
        for (var column = 0; column < result.Length; column++)
        {
            result[column] /= rms;
        }

        return result;
    }

    private static float[] ApplyScale(IReadOnlyList<float> normalizedFeatures, IReadOnlyList<float> scale)
    {
        var result = new float[normalizedFeatures.Count];
        for (var column = 0; column < result.Length; column++)
        {
            result[column] = normalizedFeatures[column] * scale[column];
        }

        return result;
    }

    private static float[,] ToSingleRowMatrix(IReadOnlyList<float> values)
    {
        var matrix = new float[1, values.Count];
        for (var column = 0; column < values.Count; column++)
        {
            matrix[0, column] = values[column];
        }

        return matrix;
    }

    private IReadOnlyList<TrainingBatch> CreateBatches(
        IReadOnlyList<BitNetTokenSequence> sequences,
        BitNetDataSplit split,
        int epochIndex)
    {
        if (sequences.Count == 0)
        {
            return [];
        }

        var batches = new List<TrainingBatch>();
        for (var index = 0; index < sequences.Count; index += _options.DataLoaderOptions.BatchSize)
        {
            var batchSize = Math.Min(_options.DataLoaderOptions.BatchSize, sequences.Count - index);
            if (batchSize < _options.DataLoaderOptions.BatchSize && _options.DataLoaderOptions.DropLast && index > 0)
            {
                break;
            }

            batches.Add(new TrainingBatch(
                split,
                sequences.Skip(index).Take(batchSize).ToArray(),
                batches.Count,
                epochIndex));
        }

        return batches;
    }

    private TrainingEvaluationSummary EvaluateSequences(IReadOnlyList<BitNetTokenSequence> sequences)
    {
        var outputWeights = _model.ExportOutputHeadWeights();
        var probabilities = new float[outputWeights.GetLength(0)];
        var totalLoss = 0d;
        var totalTokens = 0;

        foreach (var sequence in sequences)
        {
            var inputIds = sequence.TokenIds.Take(sequence.TokenIds.Count - 1).ToArray();
            var targetIds = sequence.TokenIds.Skip(1).ToArray();
            var hiddenStates = _model.ForwardHiddenStates(inputIds);

            for (var position = 0; position < targetIds.Length; position++)
            {
                var features = GetRow(hiddenStates, position);
                var logits = new float[outputWeights.GetLength(0)];
                for (var row = 0; row < outputWeights.GetLength(0); row++)
                {
                    var sum = 0f;
                    for (var column = 0; column < outputWeights.GetLength(1); column++)
                    {
                        sum += outputWeights[row, column] * features[column];
                    }

                    logits[row] = sum;
                }

                totalLoss += CrossEntropyLoss.FromLogits(logits, targetIds[position], probabilities);
                totalTokens++;
            }
        }

        var averageCrossEntropy = totalTokens == 0 ? 0d : totalLoss / totalTokens;
        return new TrainingEvaluationSummary(
            "HeldOut",
            sequences.Count,
            averageCrossEntropy,
            totalTokens == 0 ? 0d : Math.Exp(averageCrossEntropy));
    }

    private TrainingCheckpointSummary SaveCheckpoint(int epoch, int samplesSeen)
    {
        var directory = string.IsNullOrWhiteSpace(_options.CheckpointDirectory)
            ? Environment.CurrentDirectory
            : _options.CheckpointDirectory;
        Directory.CreateDirectory(directory);

        var path = Path.Combine(directory, $"{_options.CheckpointPrefix}-epoch{epoch}.bitnet.json");
        BitNetPaperCheckpoint.Save(_model, path);
        return new TrainingCheckpointSummary(epoch, samplesSeen, path);
    }

    private IReadOnlyList<TrainingEvaluationSummary> EvaluateFixtures(IReadOnlyList<BitNetBenchmarkTextFixture> fixtures) =>
        fixtures
            .Select(fixture =>
            {
                var averageCrossEntropy = CalculateAverageCrossEntropy(fixture.Samples);
                return new TrainingEvaluationSummary(
                    fixture.Name,
                    fixture.Samples.Count,
                    averageCrossEntropy,
                    averageCrossEntropy <= 0d ? 0d : Math.Exp(averageCrossEntropy));
            })
            .ToArray();

    private double CalculateAverageCrossEntropy(IReadOnlyList<string> samples)
    {
        var totalLoss = 0d;
        var totalTokens = 0;
        foreach (var sample in samples)
        {
            var tokenIds = _model.EncodeTokenIds(sample, appendEndToken: true);
            for (var index = 0; index < tokenIds.Count - 1; index++)
            {
                var context = tokenIds.Take(index + 1).ToArray();
                var logits = _model.ForwardLogits(context);
                totalLoss -= Math.Log(GetTargetProbability(logits, tokenIds[index + 1]));
                totalTokens++;
            }
        }

        return totalTokens == 0 ? 0d : totalLoss / totalTokens;
    }

    private static double GetTargetProbability(float[,] logits, int targetId)
    {
        var lastRow = logits.GetLength(0) - 1;
        var maxLogit = float.NegativeInfinity;
        for (var column = 0; column < logits.GetLength(1); column++)
        {
            maxLogit = MathF.Max(maxLogit, logits[lastRow, column]);
        }

        var partition = 0d;
        var targetProbability = 0d;
        for (var column = 0; column < logits.GetLength(1); column++)
        {
            var probabilityMass = Math.Exp(logits[lastRow, column] - maxLogit);
            partition += probabilityMass;
            if (column == targetId)
            {
                targetProbability = probabilityMass;
            }
        }

        return partition <= 0d ? 1e-9d : Math.Max(targetProbability / partition, 1e-9d);
    }

    private sealed record BatchResult(double TotalLoss, int Observations);
}
