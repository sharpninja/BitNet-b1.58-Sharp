using System.Diagnostics;
using System.Text;
using BitNetSharp.Core;
using BitNetSharp.Core.Models;
using BitNetSharp.Core.Training;
using BitNetSharp.Distributed.Contracts;
using BitNetSharp.Distributed.Coordinator.Services;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

// TruckMate SLM trainer (single-machine).
//
// Pipeline:
//   1. Generate v1 synthetic corpus deterministically (seed=42) into a
//      temp dir under the tool's working directory.
//   2. Build a BitNetTokenizer-compatible vocab from the corpus (the
//      same regex + lowercase pipeline BitNetPaperModel uses internally,
//      so the on-device tokenizer rebuild is byte-identical).
//   3. Construct a BitNetPaperModel with the small TruckMate preset
//      (dim=256, hidden=1024, layers=4, heads=8, seq=128, ~7M params).
//   4. Train one epoch on a slice of the corpus using BitNetFullTrainer
//      against (Prompt = "[USER] utterance", Response = "[INTENT] {...}")
//      pairs split out of each corpus line.
//   5. Pack the trained transformer parameters via FlatParameterPack and
//      write a TruckMateModelStore (.tmv1) checkpoint that the MAUI
//      benchmark app can load on-device.
//
// Defaults are tuned for an interactive dev-box loop (~5-15 min). Knobs
// are environment variables so we don't have to thread CLI parsing
// through this 10-day deadline crunch:
//
//   TM_OUTPUT_PATH        default: data/truckmate/truckmate-small.tmv1
//   TM_CORPUS_DIR         default: tools/TruckMateTrainer/build/corpus
//   TM_CORPUS_COUNT       default: 50000  (full v1 corpus)
//   TM_TRAIN_SUBSET       default: 5000   (lines actually fed to the trainer)
//   TM_EPOCHS             default: 1
//   TM_PRESET             default: small  (small|medium|large)
//   TM_VOCAB_CAP          default: 5174   (per scaling-truckmate-corpus doc)

var sw = Stopwatch.StartNew();

var output = Environment.GetEnvironmentVariable("TM_OUTPUT_PATH")
    ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
        "data", "truckmate", "truckmate-small.tmv1");
output = Path.GetFullPath(output);

var corpusDir = Environment.GetEnvironmentVariable("TM_CORPUS_DIR")
    ?? Path.Combine(AppContext.BaseDirectory, "build", "corpus");
corpusDir = Path.GetFullPath(corpusDir);

var totalCount = ParseInt("TM_CORPUS_COUNT", 50_000);
var trainSubset = ParseInt("TM_TRAIN_SUBSET", 5_000);
var epochs = ParseInt("TM_EPOCHS", 1);
var presetName = Environment.GetEnvironmentVariable("TM_PRESET") ?? "small";
var vocabCap = ParseInt("TM_VOCAB_CAP", 5174);

Console.WriteLine($"TruckMateTrainer");
Console.WriteLine($"  preset={presetName} epochs={epochs} corpus_count={totalCount} train_subset={trainSubset} vocab_cap={vocabCap}");
Console.WriteLine($"  corpus_dir={corpusDir}");
Console.WriteLine($"  output={output}");
Console.WriteLine();

// 1. Generate corpus (deterministic, seed=42).
var t0 = sw.Elapsed;
if (!Directory.Exists(corpusDir) || Directory.GetFiles(corpusDir, "truckmate-v1-shard-*.txt").Length == 0)
{
    Console.WriteLine($"[1/5] Generating v1 corpus ({totalCount} examples, seed=42)...");
    Directory.CreateDirectory(corpusDir);
    TruckMateCorpusGenerator.Generate(
        corpusDir,
        count: totalCount,
        examplesPerShard: 5_000,
        seed: 42,
        poolVersion: CorpusPoolVersion.V1,
        manifestName: "truckmate-v1");
}
else
{
    Console.WriteLine($"[1/5] Reusing existing corpus at {corpusDir}");
}
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 2. Read all lines, build a BitNetTokenizer-compatible vocab.
t0 = sw.Elapsed;
Console.WriteLine($"[2/5] Reading corpus + building vocab (cap={vocabCap})...");
var allLines = new List<string>(totalCount);
foreach (var shard in Directory.EnumerateFiles(corpusDir, "truckmate-v1-shard-*.txt").OrderBy(p => p))
{
    foreach (var line in File.ReadLines(shard))
    {
        if (!string.IsNullOrWhiteSpace(line))
        {
            allLines.Add(line);
        }
    }
}
Console.WriteLine($"      read {allLines.Count} lines from {corpusDir}");

// BitNetTokenizer.Tokenize lowercases + regex-splits BUT then maps
// every out-of-vocab match to <unk>. We need the raw word list to
// build the vocab in the first place, so run the same regex directly.
// Pattern is duplicated from BitNetTokenizer.TokenRegex; keep in sync.
var tokenRegex = new System.Text.RegularExpressions.Regex(
    @"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]",
    System.Text.RegularExpressions.RegexOptions.Compiled);
var freq = new Dictionary<string, int>(StringComparer.Ordinal);
foreach (var line in allLines)
{
    foreach (System.Text.RegularExpressions.Match match in tokenRegex.Matches(line.ToLowerInvariant()))
    {
        var token = match.Value;
        if (token == BitNetTokenizer.BeginToken || token == BitNetTokenizer.EndToken
            || token == BitNetTokenizer.UnknownToken)
        {
            continue;
        }
        freq.TryGetValue(token, out var c);
        freq[token] = c + 1;
    }
}
var rawVocabBudget = vocabCap - 3; // reserve for <bos>, <eos>, <unk>
var rawWords = freq
    .OrderByDescending(kvp => kvp.Value)
    .ThenBy(kvp => kvp.Key, StringComparer.Ordinal)
    .Take(rawVocabBudget)
    .Select(kvp => kvp.Key)
    .ToList();
Console.WriteLine($"      unique words={freq.Count}, kept top {rawWords.Count} (budget {rawVocabBudget})");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 3. Build BitNetPaperModel with the chosen preset shape.
t0 = sw.Elapsed;
var preset = TruckMateModelPresets.GetPreset(presetName, vocabSizeOverride: rawWords.Count + 3);
Console.WriteLine($"[3/5] {preset.ToDisplayString()}");
var cfg = new BitNetConfig(
    vocabSize: preset.VocabSize,
    dimension: preset.Dimension,
    hiddenDimension: preset.HiddenDimension,
    layerCount: preset.LayerCount,
    headCount: preset.HeadCount,
    maxSequenceLength: preset.MaxSequenceLength);
Console.WriteLine($"      head_dim={cfg.HeadDimension} kv_heads={cfg.KvHeadCount}");

var loggerFactory = NullLoggerFactory.Instance;
var paperLogger = NullLogger<BitNetPaperModel>.Instance;
var model = new BitNetPaperModel(
    new BitNetOptions(rawWords.ToArray(), VerbosityLevel.Quiet, MaxResponseTokens: 32),
    paperLogger,
    loggerFactory,
    config: cfg,
    seed: 42);
Console.WriteLine($"      model built ({model.EstimateResidentParameterBytes() / (1024d * 1024d):F1} MiB resident)");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 4. Train one epoch on a subset.
t0 = sw.Elapsed;
Console.WriteLine($"[4/5] Training {epochs} epoch(s) on {trainSubset} examples...");
var rng = new Random(42);
var subsetLines = allLines
    .OrderBy(_ => rng.Next())
    .Take(trainSubset)
    .ToList();
var trainingExamples = new List<TrainingExample>(subsetLines.Count);
foreach (var line in subsetLines)
{
    var idx = line.IndexOf("[INTENT]", StringComparison.Ordinal);
    if (idx < 0)
    {
        continue;
    }
    var prompt = line[..idx].Trim();
    var response = line[idx..].Trim();
    if (prompt.Length == 0 || response.Length == 0)
    {
        continue;
    }
    trainingExamples.Add(new TrainingExample(prompt, response));
}
Console.WriteLine($"      built {trainingExamples.Count} TrainingExample pairs");

var trainingOptions = new BitNetTrainingOptions(
    epochs: epochs,
    learningRate: 0.05f,
    dataLoaderOptions: new BitNetDataLoaderOptions(
        sequenceLength: cfg.MaxSequenceLength,
        batchSize: 1,
        validationFraction: 0d,
        testFraction: 0d,
        shuffle: false,
        dropLast: false,
        seed: 42));
var trainer = new BitNetFullTrainer(model, trainingOptions);
var report = trainer.Train(trainingExamples);
var lastLoss = report.LossHistory.Count > 0 ? report.LossHistory[report.LossHistory.Count - 1] : double.NaN;
Console.WriteLine($"      epochs done; final_loss={lastLoss:F4} ternary=(neg={report.NegativeWeights:N0} zero={report.ZeroWeights:N0} pos={report.PositiveWeights:N0})");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 5. Pack + save.
t0 = sw.Elapsed;
Console.WriteLine($"[5/5] Packing + saving {output}...");
var flat = FlatParameterPack.Pack(model.Transformer);
// Reconstruct the full vocab BitNetPaperModel uses internally so the
// on-device load can rebuild the same tokenizer + token-to-id map. The
// paper model prepends the three specials and de-dupes/lowercases the
// caller-supplied words.
var fullVocab = new List<string>(model.Config.VocabSize)
{
    BitNetTokenizer.BeginToken,
    BitNetTokenizer.EndToken,
    BitNetTokenizer.UnknownToken,
};
var seen = new HashSet<string>(StringComparer.Ordinal)
{
    BitNetTokenizer.BeginToken,
    BitNetTokenizer.EndToken,
    BitNetTokenizer.UnknownToken,
};
foreach (var word in rawWords)
{
    var lower = word.ToLowerInvariant();
    if (seen.Add(lower))
    {
        fullVocab.Add(lower);
    }
}
if (fullVocab.Count != cfg.VocabSize)
{
    Console.WriteLine($"      WARN: rebuilt vocab count {fullVocab.Count} != cfg.VocabSize {cfg.VocabSize}; padding with placeholder tokens");
    while (fullVocab.Count < cfg.VocabSize)
    {
        fullVocab.Add($"<pad{fullVocab.Count}>");
    }
}

TruckMateModelStore.Save(
    output,
    vocabSize: cfg.VocabSize,
    dimension: cfg.Dimension,
    hiddenDimension: cfg.HiddenDimension,
    layerCount: cfg.LayerCount,
    headCount: cfg.HeadCount,
    maxSequenceLength: cfg.MaxSequenceLength,
    kvHeadCount: cfg.KvHeadCount,
    vocabulary: fullVocab,
    flatParameters: flat);
var fileLen = new FileInfo(output).Length;
Console.WriteLine($"      saved {fileLen / (1024d * 1024d):F2} MiB ({flat.Length:N0} float params)");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

Console.WriteLine();
Console.WriteLine($"OK total_wall={sw.Elapsed.TotalSeconds:F1}s");
return 0;

static int ParseInt(string envName, int fallback)
{
    var raw = Environment.GetEnvironmentVariable(envName);
    return int.TryParse(raw, out var n) ? n : fallback;
}
