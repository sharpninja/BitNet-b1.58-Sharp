using System.Diagnostics;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Quantization;

namespace BitNetSharp.Core.Training;

/// <summary>
/// Full-backprop trainer that updates every BitLinear layer in a
/// <see cref="BitNetTransformer"/> plus the token-embedding matrix using the
/// Straight-Through Estimator (STE). Relies on
/// <see cref="BitNetTransformer.Backward"/> to chain gradients from logits all
/// the way back to the embedding table.
/// </summary>
public sealed class BitNetFullTrainer
{
    private readonly BitNetPaperModel? _model;
    private readonly BitNetTransformer _transformer;
    private readonly BitNetTrainingOptions _options;
    private readonly BitNetDataLoader? _loader;
    private readonly AdamWOptimizer _optimizer;
    private readonly List<(BitLinear Layer, AdamWOptimizer.OptimizerState State)> _layerStates;
    private readonly AdamWOptimizer.OptimizerState _embeddingState;

    /// <summary>
    /// Constructs a trainer that drives a <see cref="BitNetPaperModel"/> end-to-end via
    /// <see cref="Train(IEnumerable{TrainingExample})"/>. Compatible with existing callers
    /// that feed prompt/response pairs through the data loader.
    /// </summary>
    public BitNetFullTrainer(BitNetPaperModel model, BitNetTrainingOptions? options = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _transformer = model.Transformer;
        _options = options ?? new BitNetTrainingOptions(
            dataLoaderOptions: new BitNetDataLoaderOptions(
                sequenceLength: model.Config.MaxSequenceLength));
        _loader = new BitNetDataLoader(model.Options.Vocabulary, _options.DataLoaderOptions);
        _optimizer = new AdamWOptimizer(
            _options.LearningRate,
            _options.Beta1,
            _options.Beta2,
            _options.Epsilon,
            _options.WeightDecay);

        _layerStates = [];
        InitializeAllLayers();
        _embeddingState = _optimizer.CreateState(_transformer.Config.VocabSize, _transformer.Config.Dimension);
    }

    /// <summary>
    /// Constructs a trainer that operates directly on a <see cref="BitNetTransformer"/>.
    /// Used by tests and the distributed worker path where tokenized sequences are
    /// already available and the full paper-model scaffolding is unnecessary.
    /// </summary>
    public BitNetFullTrainer(BitNetTransformer transformer, BitNetTrainingOptions? options = null)
    {
        _model = null;
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _options = options ?? new BitNetTrainingOptions(
            dataLoaderOptions: new BitNetDataLoaderOptions(
                sequenceLength: transformer.Config.MaxSequenceLength));
        _loader = null;
        _optimizer = new AdamWOptimizer(
            _options.LearningRate,
            _options.Beta1,
            _options.Beta2,
            _options.Epsilon,
            _options.WeightDecay);

        _layerStates = [];
        InitializeAllLayers();
        _embeddingState = _optimizer.CreateState(_transformer.Config.VocabSize, _transformer.Config.Dimension);
    }

    public BitNetTrainingOptions Options => _options;

    /// <summary>
    /// Trains the underlying <see cref="BitNetPaperModel"/> on the supplied examples.
    /// Only available on the paper-model constructor path.
    /// </summary>
    public TrainingReport Train(IEnumerable<TrainingExample> examples)
    {
        ArgumentNullException.ThrowIfNull(examples);

        if (_model is null || _loader is null)
        {
            throw new InvalidOperationException(
                "This trainer was constructed without a BitNetPaperModel. Use Train(IReadOnlyList<int[]>, int) instead.");
        }

        var trainingSet = examples.ToList();
        if (trainingSet.Count == 0)
        {
            throw new ArgumentException("At least one training example is required.", nameof(examples));
        }

        var splitSequences = _loader.Load(trainingSet);
        var trainingSequences = splitSequences[BitNetDataSplit.Training];

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
        }

        if (trainingSequences.Count == 0)
        {
            throw new InvalidOperationException("The configured data split produced zero training sequences.");
        }

        var rawTokenSequences = trainingSequences
            .Select(sequence => sequence.TokenIds.ToArray())
            .ToArray();

        return TrainCore(rawTokenSequences, _options.Epochs, dataset: _options.TrainingDatasetName);
    }

    /// <summary>
    /// Trains the underlying <see cref="BitNetTransformer"/> on the supplied pre-tokenized
    /// sequences for the requested number of epochs.
    /// </summary>
    public TrainingReport Train(IReadOnlyList<int[]> tokenSequences, int epochs)
    {
        ArgumentNullException.ThrowIfNull(tokenSequences);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);

        if (tokenSequences.Count == 0)
        {
            throw new ArgumentException("At least one token sequence is required.", nameof(tokenSequences));
        }

        return TrainCore(tokenSequences, epochs, dataset: _options.TrainingDatasetName);
    }

    private TrainingReport TrainCore(IReadOnlyList<int[]> tokenSequences, int epochs, string? dataset)
    {
        var lossHistory = new List<double>(epochs);
        var epochMetrics = new List<TrainingEpochMetrics>(epochs);
        var totalSamples = 0;

        for (var epoch = 0; epoch < epochs; epoch++)
        {
            var epochStart = Stopwatch.GetTimestamp();
            var epochLoss = 0d;
            var epochObservations = 0;

            foreach (var sequence in tokenSequences)
            {
                if (sequence.Length < 2)
                {
                    continue;
                }

                var batchResult = TrainOnSequence(sequence);
                epochLoss += batchResult.Loss;
                epochObservations += batchResult.Observations;
                totalSamples++;
            }

            var averageLoss = epochObservations == 0 ? 0d : epochLoss / epochObservations;
            lossHistory.Add(averageLoss);
            epochMetrics.Add(new TrainingEpochMetrics(
                epoch + 1, averageLoss, totalSamples, epochObservations));

            var elapsedSeconds = Stopwatch.GetElapsedTime(epochStart).TotalSeconds;
            var perplexityHint = averageLoss > 0 ? Math.Exp(averageLoss) : double.NaN;
            Console.WriteLine(
                $"[FullTrainer] Epoch {epoch + 1}/{epochs} | Loss: {averageLoss:F6} | Perplexity: {perplexityHint:F2} | Sequences: {totalSamples:N0} | Tokens: {epochObservations:N0} | Wall: {elapsedSeconds:F2}s");
        }

        var stats = _model is null
            ? GetTransformerTernaryStats()
            : _model.GetTernaryWeightStats();

        return new TrainingReport(
            lossHistory,
            totalSamples,
            epochs,
            stats.NegativeCount,
            stats.ZeroCount,
            stats.PositiveCount,
            epochMetrics,
            [],
            [],
            dataset,
            null);
    }

    private TernaryWeightStats GetTransformerTernaryStats()
    {
        var negative = 0;
        var zero = 0;
        var positive = 0;

        foreach (var layer in _transformer.EnumerateBitLinearLayers())
        {
            var stats = layer.GetTernaryStats();
            negative += stats.NegativeCount;
            zero += stats.ZeroCount;
            positive += stats.PositiveCount;
        }

        return new TernaryWeightStats(negative, zero, positive);
    }

    private void InitializeAllLayers()
    {
        foreach (var layer in _transformer.EnumerateBitLinearLayers())
        {
            layer.InitializeMasterWeights();
            var outDim = layer.Config.OutputDimension;
            var inDim = layer.Config.InputDimension;
            var state = _optimizer.CreateState(outDim, inDim);
            _layerStates.Add((layer, state));
        }
    }

    private SequenceResult TrainOnSequence(int[] tokenIds)
    {
        var inputIds = new int[tokenIds.Length - 1];
        var targetIds = new int[tokenIds.Length - 1];
        Array.Copy(tokenIds, 0, inputIds, 0, inputIds.Length);
        Array.Copy(tokenIds, 1, targetIds, 0, targetIds.Length);

        // Full forward pass through the transformer
        var logits = _transformer.Forward(inputIds);

        // Compute loss and output gradient
        var totalLoss = 0d;
        var vocabSize = _transformer.Config.VocabSize;
        var seqLen = targetIds.Length;
        var gradLogits = new float[seqLen, vocabSize];
        var probabilities = new float[vocabSize];
        var positionLogits = new float[vocabSize];

        for (var position = 0; position < seqLen; position++)
        {
            for (var v = 0; v < vocabSize; v++)
            {
                positionLogits[v] = logits[position, v];
            }

            totalLoss += CrossEntropyLoss.FromLogits(positionLogits, targetIds[position], probabilities);

            // Gradient of cross-entropy w.r.t. logits: softmax(logits) - one_hot(target)
            for (var v = 0; v < vocabSize; v++)
            {
                gradLogits[position, v] = probabilities[v] - (v == targetIds[position] ? 1f : 0f);
            }
        }

        // Zero every gradient buffer before the backward pass.
        foreach (var (layer, _) in _layerStates)
        {
            layer.ZeroGradients();
        }
        _transformer.ZeroTokenEmbeddingGradients();

        // Backward pass drives gradients all the way to the token embeddings.
        _transformer.Backward(gradLogits);

        // Optimizer step for every BitLinear layer.
        foreach (var (layer, state) in _layerStates)
        {
            var masterWeights = layer.ExportMasterWeights();
            var masterGradients = layer.ExportMasterGradients();

            if (masterWeights is null || masterGradients is null)
            {
                continue;
            }

            var weights2D = WrapAs2D(masterWeights, layer.Config.OutputDimension, layer.Config.InputDimension);
            var gradients2D = WrapAs2D(masterGradients, layer.Config.OutputDimension, layer.Config.InputDimension);

            _optimizer.Step(weights2D, gradients2D, state);

            var updatedWeights = Flatten(weights2D);
            layer.ImportMasterWeights(updatedWeights);
            layer.SyncTernaryFromMaster();
        }

        // Optimizer step for the token embedding matrix. AdamW.Step mutates the
        // parameter tensor in place; since the transformer only exposes an
        // additive apply-update path, we clone the current embeddings, let the
        // optimizer update the clone, and hand the delta to
        // ApplyTokenEmbeddingUpdate.
        var embeddingGrads = _transformer.ExportTokenEmbeddingGradients();
        if (embeddingGrads is not null)
        {
            var currentEmbeddings = _transformer.ExportTokenEmbeddings();
            var updatedEmbeddings = (float[,])currentEmbeddings.Clone();
            _optimizer.Step(updatedEmbeddings, embeddingGrads, _embeddingState);

            var rows = updatedEmbeddings.GetLength(0);
            var cols = updatedEmbeddings.GetLength(1);
            var delta = new float[rows, cols];
            for (var r = 0; r < rows; r++)
            {
                for (var c = 0; c < cols; c++)
                {
                    delta[r, c] = updatedEmbeddings[r, c] - currentEmbeddings[r, c];
                }
            }

            _transformer.ApplyTokenEmbeddingUpdate(delta);
        }

        return new SequenceResult(totalLoss, seqLen);
    }

    private static float[,] WrapAs2D(float[] flat, int rows, int cols)
    {
        var result = new float[rows, cols];
        Buffer.BlockCopy(flat, 0, result, 0, flat.Length * sizeof(float));
        return result;
    }

    private static float[] Flatten(float[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        var result = new float[rows * cols];
        Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));
        return result;
    }

    private sealed record SequenceResult(double Loss, int Observations);
}
