using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Core.Training;

/// <summary>
/// Full-backprop trainer that updates all BitLinear layers in the transformer
/// using the Straight-Through Estimator (STE). Unlike BitNetPaperTrainer which
/// only trains the output head and final norm, this trainer updates all 7N+1
/// BitLinear projections.
/// </summary>
public sealed class BitNetFullTrainer
{
    private readonly BitNetPaperModel _model;
    private readonly BitNetTrainingOptions _options;
    private readonly BitNetDataLoader _loader;
    private readonly AdamWOptimizer _optimizer;
    private readonly List<(BitLinear Layer, AdamWOptimizer.OptimizerState State)> _layerStates;

    public BitNetFullTrainer(BitNetPaperModel model, BitNetTrainingOptions? options = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
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

        var lossHistory = new List<double>(_options.Epochs);
        var epochMetrics = new List<TrainingEpochMetrics>(_options.Epochs);
        var totalSamples = 0;

        for (var epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var epochLoss = 0d;
            var epochObservations = 0;

            foreach (var sequence in trainingSequences)
            {
                var batchResult = TrainOnSequence(sequence);
                epochLoss += batchResult.Loss;
                epochObservations += batchResult.Observations;
                totalSamples++;
            }

            var averageLoss = epochObservations == 0 ? 0d : epochLoss / epochObservations;
            lossHistory.Add(averageLoss);
            epochMetrics.Add(new TrainingEpochMetrics(
                epoch + 1, averageLoss, totalSamples, epochObservations));

            var perplexityHint = averageLoss > 0 ? Math.Exp(averageLoss) : double.NaN;
            Console.WriteLine(
                $"[FullTrainer] Epoch {epoch + 1}/{_options.Epochs} | Loss: {averageLoss:F6} | Perplexity: {perplexityHint:F2} | Sequences: {totalSamples:N0} | Tokens: {epochObservations:N0}");
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
            [],
            [],
            _options.TrainingDatasetName,
            null);
    }

    private void InitializeAllLayers()
    {
        foreach (var layer in EnumerateAllBitLinearLayers())
        {
            layer.InitializeMasterWeights();
            var outDim = layer.Config.OutputDimension;
            var inDim = layer.Config.InputDimension;
            var state = _optimizer.CreateState(outDim, inDim);
            _layerStates.Add((layer, state));
        }
    }

    private SequenceResult TrainOnSequence(BitNetTokenSequence sequence)
    {
        var inputIds = sequence.TokenIds.Take(sequence.TokenIds.Count - 1).ToArray();
        var targetIds = sequence.TokenIds.Skip(1).ToArray();

        // Full forward pass through the transformer
        var logits = _model.Transformer.Forward(inputIds);

        // Compute loss and output gradient
        var totalLoss = 0d;
        var vocabSize = _model.Config.VocabSize;
        var seqLen = targetIds.Length;
        var gradLogits = new float[seqLen, vocabSize];
        var probabilities = new float[vocabSize];

        for (var position = 0; position < seqLen; position++)
        {
            // Extract logits for this position
            var positionLogits = new float[vocabSize];
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

        // Zero all gradients
        foreach (var (layer, _) in _layerStates)
        {
            layer.ZeroGradients();
        }

        // Backward pass through the entire model
        // OutputHead backward
        var gradHidden = _model.Transformer.OutputHead.BackwardSTE(gradLogits);

        // FinalNorm backward
        gradHidden = _model.Transformer.FinalNorm.BackwardSTE(gradHidden);

        // Backward through transformer layers in reverse order
        for (var layerIndex = _model.Transformer.Layers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            gradHidden = _model.Transformer.Layers[layerIndex].BackwardSTE(gradHidden);
        }

        // Update all master weights
        foreach (var (layer, state) in _layerStates)
        {
            var masterWeights = layer.ExportMasterWeights();
            var masterGradients = layer.ExportMasterGradients();

            if (masterWeights is null || masterGradients is null)
            {
                continue;
            }

            // Wrap flat arrays as single-row 2D for optimizer compatibility
            var weights2D = WrapAs2D(masterWeights, layer.Config.OutputDimension, layer.Config.InputDimension);
            var gradients2D = WrapAs2D(masterGradients, layer.Config.OutputDimension, layer.Config.InputDimension);

            _optimizer.Step(weights2D, gradients2D, state);

            // Import updated weights back and re-quantize
            var updatedWeights = Flatten(weights2D);
            layer.ImportMasterWeights(updatedWeights);
            layer.SyncTernaryFromMaster();
        }

        return new SequenceResult(totalLoss, seqLen);
    }

    private IEnumerable<BitLinear> EnumerateAllBitLinearLayers()
    {
        foreach (var layer in _model.Transformer.Layers)
        {
            yield return layer.Attention.QueryProjection;
            yield return layer.Attention.KeyProjection;
            yield return layer.Attention.ValueProjection;
            yield return layer.Attention.OutputProjection;
            yield return layer.FeedForward.GateProjection;
            yield return layer.FeedForward.UpProjection;
            yield return layer.FeedForward.DownProjection;
        }

        yield return _model.Transformer.OutputHead;
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
