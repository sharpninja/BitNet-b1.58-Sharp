using System;
using BitNetSharp.Core.Models;

namespace BitNetSharp.Converter.Tests;

public sealed class BitNetConfigQwen3PresetTests
{
    [Fact]
    public void Qwen3Like8B_ReturnsConfigWithExpectedShape()
    {
        var config = BitNetConfig.Qwen3Like8B(vocabSize: 5174);

        Assert.Equal(5174, config.VocabSize);
        Assert.Equal(4096, config.Dimension);
        Assert.Equal(12288, config.HiddenDimension);
        Assert.Equal(36, config.LayerCount);
        Assert.Equal(32, config.HeadCount);
        Assert.Equal(8, config.KvHeadCount);
        Assert.Equal(65536, config.MaxSequenceLength);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(1_000_000f, config.RopeTheta);
    }

    [Fact]
    public void Default_KvHeadCount_EqualsHeadCount_ForBackCompat()
    {
        var defaultConfig = new BitNetConfig();
        Assert.Equal(defaultConfig.HeadCount, defaultConfig.KvHeadCount);
    }

    [Fact]
    public void Default_RopeTheta_IsTenThousand()
    {
        var defaultConfig = new BitNetConfig();
        Assert.Equal(10_000f, defaultConfig.RopeTheta);
    }

    [Fact]
    public void Constructor_RejectsKvHeadCount_WhenHeadCountNotDivisible()
    {
        // HeadCount=32, KvHeadCount=7 -> 32 % 7 != 0
        Assert.Throws<ArgumentException>(() => new BitNetConfig(
            vocabSize: 1024,
            dimension: 512,
            hiddenDimension: 2048,
            layerCount: 4,
            headCount: 32,
            maxSequenceLength: 256,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 7,
            ropeTheta: 10_000f));
    }

    [Fact]
    public void Constructor_RejectsZeroKvHeadCount()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BitNetConfig(
            vocabSize: 1024,
            dimension: 512,
            hiddenDimension: 2048,
            layerCount: 4,
            headCount: 8,
            maxSequenceLength: 256,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 0,
            ropeTheta: 10_000f));
    }

    [Fact]
    public void Constructor_RejectsZeroRopeTheta()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BitNetConfig(
            vocabSize: 1024,
            dimension: 512,
            hiddenDimension: 2048,
            layerCount: 4,
            headCount: 8,
            maxSequenceLength: 256,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 8,
            ropeTheta: 0f));
    }

    [Fact]
    public void Constructor_AcceptsValidGqaRatio()
    {
        var config = new BitNetConfig(
            vocabSize: 1024,
            dimension: 512,
            hiddenDimension: 2048,
            layerCount: 4,
            headCount: 32,
            maxSequenceLength: 256,
            rmsNormEpsilon: 1e-5f,
            kvHeadCount: 8,
            ropeTheta: 1_000_000f);

        Assert.Equal(32, config.HeadCount);
        Assert.Equal(8, config.KvHeadCount);
        Assert.Equal(1_000_000f, config.RopeTheta);
    }
}
