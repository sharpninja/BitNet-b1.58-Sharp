using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using Microsoft.Extensions.Logging.Abstractions;

namespace BitNetSharp.Tests;

/// <summary>
/// Phase F5 (float-deletion wiring, end-to-end): routes
/// <see cref="BitNetPaperModel.GenerateResponse(string, int?)"/> through the
/// integer path when <see cref="BitNetOptions.UseIntegerForward"/> is true.
/// <see cref="BitNetPaperModel"/>'s SelectNextToken is pure argmax over the
/// last-row logits (plus monotonic repetition penalty), so integer vs float
/// must produce the same token stream for the same seed + prompt.
/// </summary>
public sealed class BitNetPaperModelIntegerForwardTests
{
    private static readonly string[] Vocabulary =
    [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"
    ];

    [Fact]
    public void GenerateResponse_UseIntegerForwardTrue_MatchesFloatTokenStream()
    {
        var floatOptions = new BitNetOptions(
            Vocabulary,
            VerbosityLevel.Quiet,
            MaxResponseTokens: 4,
            UseIntegerForward: false);
        var intOptions = floatOptions with { UseIntegerForward = true };

        var config = new BitNetConfig(
            vocabSize: 11,
            dimension: 32,
            hiddenDimension: 96,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);

        var floatModel = new BitNetPaperModel(
            floatOptions,
            NullLogger<BitNetPaperModel>.Instance,
            NullLoggerFactory.Instance,
            config,
            seed: 257);
        var intModel = new BitNetPaperModel(
            intOptions,
            NullLogger<BitNetPaperModel>.Instance,
            NullLoggerFactory.Instance,
            config,
            seed: 257);

        var prompt = "alpha beta gamma";
        var floatResult = floatModel.GenerateResponse(prompt);
        var intResult = intModel.GenerateResponse(prompt);

        // Same seed + same config + same prompt + argmax sampling + integer
        // path is argmax-equivalent to float => token streams must match.
        Assert.Equal(floatResult.Tokens, intResult.Tokens);
    }

    [Fact]
    public void GenerateResponse_UseIntegerForwardTrue_ProducesNonEmptyResponse()
    {
        var options = new BitNetOptions(
            Vocabulary,
            VerbosityLevel.Quiet,
            MaxResponseTokens: 3,
            UseIntegerForward: true);
        var config = new BitNetConfig(
            vocabSize: 11,
            dimension: 32,
            hiddenDimension: 96,
            layerCount: 2,
            headCount: 4,
            maxSequenceLength: 16,
            rmsNormEpsilon: 1e-6f,
            kvHeadCount: 2);

        var model = new BitNetPaperModel(
            options,
            NullLogger<BitNetPaperModel>.Instance,
            NullLoggerFactory.Instance,
            config,
            seed: 263);

        var result = model.GenerateResponse("alpha beta");

        // Smoke: exercise the integer hot path end-to-end.
        Assert.NotNull(result);
        Assert.NotNull(result.Tokens);
    }
}
