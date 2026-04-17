using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;

namespace BitNetSharp.Tests;

/// <summary>
/// Unit tests for <see cref="FlatParameterPack"/>. Covers the
/// round-trip identity and the computed length invariant that the
/// distributed training protocol relies on.
/// </summary>
public sealed class FlatParameterPackTests
{
    private static BitNetConfig TinyConfig() => new(
        vocabSize: 32,
        dimension: 16,
        hiddenDimension: 32,
        layerCount: 2,
        headCount: 2,
        maxSequenceLength: 16);

    [Fact]
    public void ComputeLength_matches_token_embeddings_plus_bitlinear_sizes()
    {
        var cfg = TinyConfig();

        // Token embeddings
        long expected = (long)cfg.VocabSize * cfg.Dimension;

        // Per layer: Q, K, V, O (dim*dim), Gate/Up (dim*hidden), Down (hidden*dim)
        long perLayer =
            4L * cfg.Dimension * cfg.Dimension
            + 2L * cfg.Dimension * cfg.HiddenDimension
            + 1L * cfg.HiddenDimension * cfg.Dimension;
        expected += cfg.LayerCount * perLayer;

        // OutputHead (dim -> vocab)
        expected += (long)cfg.Dimension * cfg.VocabSize;

        Assert.Equal(expected, FlatParameterPack.ComputeLength(cfg));
    }

    [Fact]
    public void Pack_returns_vector_of_computed_length_for_fresh_model()
    {
        var cfg = TinyConfig();
        var transformer = new BitNetTransformer(cfg, seed: 7);

        var flat = FlatParameterPack.Pack(transformer);

        Assert.Equal(FlatParameterPack.ComputeLength(cfg), flat.Length);
    }

    [Fact]
    public void Pack_Unpack_round_trip_preserves_flat_vector()
    {
        var cfg = TinyConfig();
        var transformer = new BitNetTransformer(cfg, seed: 13);

        var original = FlatParameterPack.Pack(transformer);

        // Perturb then round-trip.
        var mutated = new float[original.Length];
        for (var i = 0; i < mutated.Length; i++)
        {
            mutated[i] = original[i] + 0.01f * (i % 7 - 3);
        }

        FlatParameterPack.Unpack(transformer, mutated);
        var repacked = FlatParameterPack.Pack(transformer);

        Assert.Equal(mutated.Length, repacked.Length);
        for (var i = 0; i < mutated.Length; i++)
        {
            Assert.Equal(mutated[i], repacked[i]);
        }
    }

    [Fact]
    public void Unpack_rejects_flat_vector_of_wrong_length()
    {
        var cfg = TinyConfig();
        var transformer = new BitNetTransformer(cfg, seed: 3);
        var wrong = new float[FlatParameterPack.ComputeLength(cfg) + 5];

        var ex = Assert.Throws<ArgumentException>(() => FlatParameterPack.Unpack(transformer, wrong));
        Assert.Contains("does not match expected", ex.Message);
    }

    /// <summary>
    /// Track 7: locks in the canonical flat-parameter-vector length
    /// for the TruckMate "small" preset that the coordinator serves
    /// and the worker trains against. If this number changes, the
    /// coordinator's on-disk weights and every worker's
    /// BITNET_MODEL_PRESET wiring must move together — so flip the
    /// value here deliberately if the preset shape actually changed.
    /// </summary>
    [Fact]
    public void ComputeLength_for_small_preset_is_canonical_6_843_392()
    {
        var preset = TruckMateModelPresets.Small();
        var cfg = new BitNetConfig(
            vocabSize: preset.VocabSize,
            dimension: preset.Dimension,
            hiddenDimension: preset.HiddenDimension,
            layerCount: preset.LayerCount,
            headCount: preset.HeadCount,
            maxSequenceLength: preset.MaxSequenceLength);

        var flatLength = FlatParameterPack.ComputeLength(cfg);

        Assert.Equal(6_843_392, flatLength);
    }

    [Fact]
    public void Packed_vector_contains_no_nan_or_inf()
    {
        var cfg = TinyConfig();
        var transformer = new BitNetTransformer(cfg, seed: 21);

        var flat = FlatParameterPack.Pack(transformer);

        for (var i = 0; i < flat.Length; i++)
        {
            Assert.False(float.IsNaN(flat[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(flat[i]), $"Inf at index {i}");
        }
    }
}
