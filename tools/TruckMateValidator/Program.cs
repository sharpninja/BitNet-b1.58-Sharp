using System.Diagnostics;
using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using Microsoft.Extensions.Logging.Abstractions;

// Validator: loads a truckmate-small.tmv1 checkpoint via TruckMateModelStore,
// reconstructs a BitNetTransformer + WordLevelTokenizer, wraps in
// WordLevelInferenceModel, and runs a fixed bank of TruckMate intent
// prompts through StreamGenerateAsync. Reports per-prompt TTFT,
// per-decode-token ms, throughput, and intent-substring accuracy.
//
// Mirrors the MAUI IntentBenchPage; provides x86 sanity check + CI
// regression guard before APK install.
//
// Usage:
//   dotnet run -c Release --project tools/TruckMateValidator [path/to/.tmv1]

var path = args.Length > 0
    ? args[0]
    : Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
        "data", "truckmate", "truckmate-small.tmv1");
path = Path.GetFullPath(path);

if (!File.Exists(path))
{
    Console.Error.WriteLine($"FAIL: checkpoint not found at {path}");
    Console.Error.WriteLine("Run tools/TruckMateTrainer first to produce one.");
    return 2;
}

Console.WriteLine($"TruckMateValidator (WordLevel pipeline)");
Console.WriteLine($"  checkpoint: {path}");
Console.WriteLine($"  size: {new FileInfo(path).Length:N0} bytes");
Console.WriteLine();

var sw = Stopwatch.StartNew();
var (header, flat) = TruckMateModelStore.Load(path);
Console.WriteLine($"[1/3] Loaded TMV1 in {sw.Elapsed.TotalMilliseconds:F0} ms");
Console.WriteLine($"      vocab={header.VocabSize} dim={header.Dimension} hidden={header.HiddenDimension}");
Console.WriteLine($"      layers={header.LayerCount} heads={header.HeadCount} kvHeads={header.KvHeadCount} headDim={header.Dimension / header.HeadCount}");
Console.WriteLine($"      seq_max={header.MaxSequenceLength} flat_floats={flat.Length:N0}");

sw.Restart();
var cfg = new BitNetConfig(
    vocabSize: header.VocabSize,
    dimension: header.Dimension,
    hiddenDimension: header.HiddenDimension,
    layerCount: header.LayerCount,
    headCount: header.HeadCount,
    maxSequenceLength: header.MaxSequenceLength,
    kvHeadCount: header.KvHeadCount);

// Build transformer + import flat weights.
var transformer = new BitNetTransformer(
    cfg,
    NullLogger<BitNetTransformer>.Instance,
    seed: 42,
    skipRandomInit: true);
FlatParameterPack.Unpack(transformer, flat);

// Reconstruct tokenizer from the saved vocab list. SaveToFile + LoadFromFile
// is the canonical round-trip path used by coordinator/worker; here we
// build a temp file in memory.
var vocabJson = System.Text.Json.JsonSerializer.Serialize(header.Vocabulary.ToArray());
var tempVocabPath = Path.Combine(Path.GetTempPath(), $"tmvocab-{Guid.NewGuid():N}.json");
File.WriteAllText(tempVocabPath, vocabJson, System.Text.Encoding.UTF8);
WordLevelTokenizer tokenizer;
try
{
    tokenizer = WordLevelTokenizer.LoadFromFile(tempVocabPath);
}
finally
{
    try { File.Delete(tempVocabPath); } catch { /* best-effort */ }
}

var model = new WordLevelInferenceModel(transformer, tokenizer)
{
    MaxResponseTokens = 24,
    SuppressEosAndUnk = true, // smoke build will argmax to EOS quickly otherwise
};
Console.WriteLine($"[2/3] Built WordLevelInferenceModel in {sw.Elapsed.TotalMilliseconds:F0} ms");
Console.WriteLine($"      resident_bytes={transformer.EstimateResidentParameterBytes():N0}");
Console.WriteLine();

(string Utterance, string ExpectedIntent)[] prompts =
{
    ("[USER] start my trip to dallas [INTENT]",                 "start_trip"),
    ("[USER] take me to memphis [INTENT]",                      "navigate"),
    ("[USER] find me a truck stop near nashville [INTENT]",     "find_poi"),
    ("[USER] avoid tolls [INTENT]",                             "route_preference"),
    ("[USER] check my hours [INTENT]",                          "hos_status"),
    ("[USER] what's my eta [INTENT]",                           "eta_query"),
    ("[USER] reroute around traffic [INTENT]",                  "reroute"),
    ("[USER] mark load LD-12345 as delivered [INTENT]",         "update_load"),
};

Console.WriteLine($"[3/3] Running {prompts.Length} intent prompts (max_tokens=24)...");
Console.WriteLine();
Console.WriteLine($"step | ttft_ms | tokens | per_dec_ms | hit | utterance -> response");
Console.WriteLine(new string('-', 100));

long sumTtft = 0;
int sumTokens = 0;
int hits = 0;
var perDecodeMs = new List<double>();

for (var i = 0; i < prompts.Length; i++)
{
    var (utterance, expectedIntent) = prompts[i];
    var promptSw = Stopwatch.StartNew();
    long ttftMs = 0;
    var first = true;
    var sumDecode = 0d;
    var step = 0;
    var responseText = new System.Text.StringBuilder();

    await foreach (var token in model.StreamGenerateAsync(utterance, maxTokens: 24))
    {
        if (first)
        {
            ttftMs = promptSw.ElapsedMilliseconds;
            first = false;
        }
        sumDecode += token.DecodeMs;
        responseText.Append(token.TokenText);
        step++;
    }
    promptSw.Stop();

    var perDec = step > 1 ? sumDecode / (step - 1) : 0d;
    var responseStr = responseText.ToString();
    var hit = responseStr.Contains(expectedIntent, StringComparison.OrdinalIgnoreCase);
    if (hit) hits++;
    sumTtft += ttftMs;
    sumTokens += step;
    perDecodeMs.Add(perDec);

    var snippet = responseStr.Length > 60 ? responseStr[..60] + "..." : responseStr;
    Console.WriteLine($"{i + 1,4} | {ttftMs,7} | {step,6} | {perDec,10:F1} | {(hit ? " Y " : " . ")} | {expectedIntent,-20} -> {snippet.Replace('\n', ' ')}");
}

Console.WriteLine(new string('-', 100));
Console.WriteLine($"prompts={prompts.Length} hits={hits} accuracy={(double)hits / prompts.Length * 100:F1}%");
Console.WriteLine($"avg_ttft_ms={sumTtft / (double)prompts.Length:F1}");
Console.WriteLine($"avg_tokens_per_prompt={sumTokens / (double)prompts.Length:F1}");
if (perDecodeMs.Count > 0)
{
    var sorted = perDecodeMs.OrderBy(x => x).ToList();
    var p50 = sorted[sorted.Count / 2];
    var p95 = sorted[Math.Min(sorted.Count - 1, (int)(sorted.Count * 0.95))];
    Console.WriteLine($"per_decode_ms p50={p50:F1} p95={p95:F1} max={sorted[^1]:F1}");
    Console.WriteLine($"throughput_tok_per_sec p50={(p50 > 0 ? 1000d / p50 : 0d):F2}");
}

return 0;
