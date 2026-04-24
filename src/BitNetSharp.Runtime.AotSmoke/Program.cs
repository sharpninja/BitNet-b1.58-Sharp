using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Runtime;
using Microsoft.Extensions.Logging.Abstractions;

// Phase 1d AOT publish smoke:
// 1. Build a tiny BitNetConfig.
// 2. Hydrate a BitNetPaperModel, pack its weights to a fp32 flat array.
// 3. Encode as a WeightBlobCodec v1 blob.
// 4. Load it via InProcessBitNetModel.LoadFromBytes — the exact path a
//    TruckMate-style host will take after downloading a promoted blob.
// 5. Generate one response. Exit 0 on success; nonzero on any exception.
//
// A successful `dotnet publish -c Release -r win-x64 --self-contained -p:PublishAot=true`
// of this project proves:
//   - BitNetSharp.Runtime has no AOT/trim analyzer warnings
//   - Transitive Core + Contracts deps compile + link through the trim graph
//   - InProcessBitNetModel's runtime path works after native-AOT publish

try
{
    var config = new BitNetConfig(
        vocabSize: 11,
        dimension: 16,
        hiddenDimension: 32,
        layerCount: 1,
        headCount: 2,
        maxSequenceLength: 16);

    var options = new BitNetOptions(
        Vocabulary: new[] { "hello", "world", "how", "are", "you", "ok", "fine", "model" },
        Verbosity: VerbosityLevel.Normal,
        MaxResponseTokens: 4);

    var seed = new BitNetPaperModel(options, NullLogger<BitNetPaperModel>.Instance, NullLoggerFactory.Instance, config, seed: 7);
    var flat = FlatParameterPack.Pack(seed.Transformer);
    var blob = WeightBlobCodec.Encode(version: 1L, flat);

    using var runtime = InProcessBitNetModel.LoadFromBytes(blob, options, config);
    var response = runtime.GenerateResponse("hello", maxOutputTokens: 2);

    Console.WriteLine($"OK version={runtime.WeightVersion} textLen={response.Text.Length}");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"FAIL: {ex.GetType().FullName}: {ex.Message}");
    return 1;
}
