using BitNetSharp.Core.Inference;
using BitNetSharp.Core.Layers;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F6: per-layer integer primitive cache removes the big
/// per-call cost of IntegerRotaryPositionEmbedding (sin/cos table of
/// O(maxSeq * headDim/2) Math.Sin/Cos calls), IntegerRmsNorm (scale
/// copy), and the tiny IntegerSoftmax/IntegerSwiGLU allocations.
/// Inference weights are frozen, so one construction per layer is
/// safe; ConditionalWeakTable keys the cache on layer identity so
/// it's GC-friendly for transient test models.
/// </summary>
public sealed class IntegerLayerPrimitiveCacheTests
{
    [Fact]
    public void Get_SameLayerTwice_ReturnsSameAttnRmsFfnRmsRope()
    {
        var layer = BuildLayer(seed: 271);

        var first = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq: 128);
        var second = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq: 128);

        Assert.Same(first.AttnRms, second.AttnRms);
        Assert.Same(first.FfnRms, second.FfnRms);
        Assert.Same(first.Rope, second.Rope);
    }

    [Fact]
    public void Get_DifferentLayers_ReturnsDistinctInstances()
    {
        var layerA = BuildLayer(seed: 277);
        var layerB = BuildLayer(seed: 281);

        var a = IntegerLayerPrimitiveCache.Get(layerA, ropeMaxSeq: 128);
        var b = IntegerLayerPrimitiveCache.Get(layerB, ropeMaxSeq: 128);

        Assert.NotSame(a.AttnRms, b.AttnRms);
        Assert.NotSame(a.FfnRms, b.FfnRms);
        Assert.NotSame(a.Rope, b.Rope);
    }

    [Fact]
    public void Get_GrowingRopeMaxSeq_RebuildsRopeOnly()
    {
        var layer = BuildLayer(seed: 283);

        var small = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq: 64);
        var large = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq: 256);

        // RMS norms don't depend on seq length, so they stay cached.
        Assert.Same(small.AttnRms, large.AttnRms);
        Assert.Same(small.FfnRms, large.FfnRms);

        // RoPE rebuilds when required seq grows.
        Assert.NotSame(small.Rope, large.Rope);
        Assert.Equal(256, large.Rope.MaxSequenceLength);

        // Asking for a shorter seq after growth does not rebuild.
        var shrink = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq: 32);
        Assert.Same(large.Rope, shrink.Rope);
    }

    [Fact]
    public void AttnRms_ImportsLayerPreAttentionNormScale()
    {
        var layer = BuildLayer(seed: 293);
        var expected = layer.PreAttentionNorm.ExportScale();

        var primitives = IntegerLayerPrimitiveCache.Get(layer, ropeMaxSeq: 128);
        var actual = primitives.AttnRms.ExportScale();

        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void SoftmaxAndSwiGLU_AreStatelessSingletons()
    {
        // Stateless primitives can safely be reused across layers.
        Assert.Same(IntegerLayerPrimitiveCache.Softmax, IntegerLayerPrimitiveCache.Softmax);
        Assert.Same(IntegerLayerPrimitiveCache.SwiGLU, IntegerLayerPrimitiveCache.SwiGLU);
    }

    private static BitNetLayer BuildLayer(int seed)
    {
        var config = new BitNetConfig(
            vocabSize: 32,
            dimension: 64,
            hiddenDimension: 192,
            layerCount: 1,
            headCount: 2,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);
        return new BitNetLayer(config, new Random(seed));
    }
}
