using System;
using System.Collections.Generic;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;

namespace BitNetSharp.Distributed.Worker;

/// <summary>
/// Replaces the Phase D-4b synthetic-gradient stub in the worker's
/// work loop. Given the current flat master-parameter vector and a
/// pre-tokenized shard, constructs a fresh
/// <see cref="BitNetTransformer"/>, loads the vector into it, runs
/// <see cref="BitNetFullTrainer.Train(IReadOnlyList{int[]}, int)"/>
/// for a bounded number of local epochs, then returns
/// <c>new_flat - old_flat</c> as the gradient payload the coordinator
/// will aggregate.
///
/// <para>
/// The "gradient" returned here is actually the delta between the
/// worker's locally-trained weights and the weights it was assigned.
/// For <c>K=1</c> local steps with a plain SGD optimizer this is
/// equivalent to <c>-lr * grad</c>; for larger K it is the accumulated
/// effect of K optimizer steps. The coordinator treats it opaquely.
/// </para>
///
/// <para>
/// This class is deliberately stateless: every call builds a fresh
/// transformer so there is no cross-task weight drift beyond what the
/// incoming flat vector encodes. The computation is CPU-bound and
/// allocates approximately the full model size in doubles; callers
/// should run it off the heartbeat path.
/// </para>
/// </summary>
internal static class RealTrainingGradient
{
    /// <summary>
    /// Computes <c>Pack(Train(Unpack(currentFlat))) - currentFlat</c>
    /// for the supplied shard token sequences.
    /// </summary>
    /// <param name="currentFlat">Flat master-parameter vector the
    /// coordinator sent on <c>GET /weights/{version}</c>.</param>
    /// <param name="shardTokenSequences">Pre-chunked token sequences to
    /// train on. Each inner array must be at least 2 tokens long so
    /// a single next-token prediction step is possible; shorter
    /// sequences are silently skipped by the trainer.</param>
    /// <param name="config">Model configuration matching the
    /// coordinator's global model. <see cref="FlatParameterPack.ComputeLength"/>
    /// of this config MUST equal <paramref name="currentFlat"/>.Length.</param>
    /// <param name="localSteps">Number of local epochs (K in the
    /// async-SGD literature). 1 = pure async; larger values amortize
    /// the round trip but increase staleness.</param>
    /// <param name="learningRate">Optional training learning rate. Defaults
    /// to the <see cref="BitNetTrainingOptions"/> default.</param>
    /// <param name="seed">Deterministic seed for the freshly-constructed
    /// transformer so unit tests can reproduce results.</param>
    /// <returns>Flat delta <c>new_flat - old_flat</c>, same length as
    /// <paramref name="currentFlat"/>.</returns>
    public static float[] ComputeGradient(
        float[] currentFlat,
        IReadOnlyList<int[]> shardTokenSequences,
        BitNetConfig config,
        int localSteps,
        float? learningRate = null,
        int seed = 42)
    {
        ArgumentNullException.ThrowIfNull(currentFlat);
        ArgumentNullException.ThrowIfNull(shardTokenSequences);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(localSteps);

        var expected = FlatParameterPack.ComputeLength(config);
        if (currentFlat.Length != expected)
        {
            throw new ArgumentException(
                $"Flat vector length {currentFlat.Length} does not match expected {expected} for this configuration.",
                nameof(currentFlat));
        }

        if (shardTokenSequences.Count == 0)
        {
            // No training data → zero delta. Matches the coordinator's
            // expected payload shape; a noisy warning is the worker's
            // responsibility.
            return new float[currentFlat.Length];
        }

        var transformer = new BitNetTransformer(config, seed);
        FlatParameterPack.Unpack(transformer, currentFlat);

        // Build training options. Using the transformer-constructor
        // path so BitNetFullTrainer skips the paper-model data loader.
        var options = learningRate is null
            ? new BitNetTrainingOptions(
                epochs: localSteps,
                dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: config.MaxSequenceLength))
            : new BitNetTrainingOptions(
                epochs: localSteps,
                learningRate: learningRate.Value,
                dataLoaderOptions: new BitNetDataLoaderOptions(sequenceLength: config.MaxSequenceLength));

        var trainer = new BitNetFullTrainer(transformer, options);

        // BitNetFullTrainer.Train requires sequences of at least 2 tokens.
        // Filter proactively so we don't throw on an all-short shard.
        var trainable = new List<int[]>(shardTokenSequences.Count);
        foreach (var seq in shardTokenSequences)
        {
            if (seq is not null && seq.Length >= 2)
            {
                trainable.Add(seq);
            }
        }

        if (trainable.Count == 0)
        {
            return new float[currentFlat.Length];
        }

        trainer.Train(trainable, localSteps);

        var newFlat = FlatParameterPack.Pack(transformer);

        // Delta = new - old, same length as currentFlat.
        var delta = new float[newFlat.Length];
        for (var i = 0; i < delta.Length; i++)
        {
            delta[i] = newFlat[i] - currentFlat[i];
        }

        return delta;
    }
}
