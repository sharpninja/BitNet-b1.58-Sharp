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
//   1. Generate v1 synthetic corpus deterministically (seed=42).
//   2. Train a WordLevelTokenizer over the corpus (5174 cap by default).
//      This matches what the coordinator's tokenize-corpus CLI does, so
//      checkpoints from this trainer are interchangeable with
//      coordinator-trained .tmv1 files at inference time.
//   3. Build a bare BitNetTransformer sized to the chosen preset.
//   4. Pre-tokenize each corpus line into int[] sequences and call
//      BitNetFullTrainer(transformer, options).Train(sequences, epochs).
//   5. Pack the trained parameters via FlatParameterPack and write a
//      TruckMateModelStore (.tmv1) checkpoint that the MAUI app loads
//      via WordLevelInferenceModel.
//
// Knobs (env vars):
//   TM_OUTPUT_PATH       default: data/truckmate/truckmate-small.tmv1
//   TM_CORPUS_DIR        default: tools/TruckMateTrainer/build/corpus
//   TM_CORPUS_COUNT      default: 50000  (full v1 corpus)
//   TM_TRAIN_SUBSET      default: 1000   (lines fed to the trainer)
//   TM_SUBSET_OFFSET     default: 0      (start index for the subset window)
//   TM_EPOCHS            default: 1
//   TM_PRESET            default: small  (small|medium|large)
//   TM_VOCAB_CAP         default: 5174   (per scaling-truckmate-corpus doc)

var sw = Stopwatch.StartNew();

var output = Environment.GetEnvironmentVariable("TM_OUTPUT_PATH")
    ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..",
        "data", "truckmate", "truckmate-small.tmv1");
output = Path.GetFullPath(output);

var corpusDir = Environment.GetEnvironmentVariable("TM_CORPUS_DIR")
    ?? Path.Combine(AppContext.BaseDirectory, "build", "corpus");
corpusDir = Path.GetFullPath(corpusDir);

var totalCount = ParseInt("TM_CORPUS_COUNT", 50_000);
var trainSubset = ParseInt("TM_TRAIN_SUBSET", 1_000);
var subsetOffset = ParseInt("TM_SUBSET_OFFSET", 0);
var epochs = ParseInt("TM_EPOCHS", 1);
var presetName = Environment.GetEnvironmentVariable("TM_PRESET") ?? "small";
var vocabCap = ParseInt("TM_VOCAB_CAP", 5174);

Console.WriteLine($"TruckMateTrainer (WordLevel pipeline)");
Console.WriteLine($"  preset={presetName} epochs={epochs} corpus_count={totalCount}");
Console.WriteLine($"  subset_offset={subsetOffset} train_subset={trainSubset} vocab_cap={vocabCap}");
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

// 2. Read all lines, train WordLevelTokenizer.
t0 = sw.Elapsed;
Console.WriteLine($"[2/5] Reading corpus + training WordLevelTokenizer (cap={vocabCap})...");
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

var tokenizer = WordLevelTokenizer.TrainFromCorpus(allLines, maxVocab: vocabCap, minFrequency: 1);
if (tokenizer.VocabSize > vocabCap)
{
    Console.Error.WriteLine($"      Vocab size {tokenizer.VocabSize} exceeds cap {vocabCap}; aborting.");
    return 3;
}
Console.WriteLine($"      vocab_size={tokenizer.VocabSize} (specials at 0..5: PAD/UNK/BOS/EOS/USER/INTENT)");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 3. Build BitNetTransformer with the chosen preset.
t0 = sw.Elapsed;
var preset = TruckMateModelPresets.GetPreset(presetName, vocabSizeOverride: tokenizer.VocabSize);
Console.WriteLine($"[3/5] {preset.ToDisplayString()}");
var cfg = new BitNetConfig(
    vocabSize: preset.VocabSize,
    dimension: preset.Dimension,
    hiddenDimension: preset.HiddenDimension,
    layerCount: preset.LayerCount,
    headCount: preset.HeadCount,
    maxSequenceLength: preset.MaxSequenceLength);
Console.WriteLine($"      head_dim={cfg.HeadDimension} kv_heads={cfg.KvHeadCount}");

var transformer = new BitNetTransformer(cfg, NullLogger<BitNetTransformer>.Instance, seed: 42);
Console.WriteLine($"      transformer built ({transformer.EstimateResidentParameterBytes() / (1024d * 1024d):F1} MiB resident)");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 4. Tokenize the chosen subset slice + train.
t0 = sw.Elapsed;
Console.WriteLine($"[4/5] Tokenizing + training {epochs} epoch(s) on lines [{subsetOffset}..{subsetOffset + trainSubset})...");
if (subsetOffset >= allLines.Count)
{
    Console.Error.WriteLine($"      subset_offset {subsetOffset} >= corpus size {allLines.Count}");
    return 4;
}
var sliceEnd = Math.Min(allLines.Count, subsetOffset + trainSubset);
var subsetLines = allLines.GetRange(subsetOffset, sliceEnd - subsetOffset);

var tokenSequences = new List<int[]>(subsetLines.Count);
var totalTokensSeen = 0L;
foreach (var line in subsetLines)
{
    // Use Encode here (BOS + tokens + EOS) so the trainer sees a complete
    // utterance with terminal EOS — mirrors what the coordinator's
    // tokenize-corpus produces in the .bin shards.
    var ids = tokenizer.Encode(line);
    if (ids.Length < 2) continue;
    if (ids.Length > cfg.MaxSequenceLength)
    {
        // Truncate from the front, preserving BOS at index 0 and EOS at -1.
        var keep = cfg.MaxSequenceLength;
        var trimmed = new int[keep];
        trimmed[0] = WordLevelTokenizer.BosId;
        trimmed[^1] = WordLevelTokenizer.EosId;
        Array.Copy(ids, ids.Length - keep + 1, trimmed, 1, keep - 2);
        ids = trimmed;
    }
    tokenSequences.Add(ids);
    totalTokensSeen += ids.Length;
}
Console.WriteLine($"      tokenized {tokenSequences.Count} sequences, total_tokens={totalTokensSeen:N0}");

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
var trainer = new BitNetFullTrainer(transformer, trainingOptions);
var report = trainer.Train(tokenSequences, epochs);
var lastLoss = report.LossHistory.Count > 0 ? report.LossHistory[report.LossHistory.Count - 1] : double.NaN;
Console.WriteLine($"      epochs done; final_loss={lastLoss:F4} ternary=(neg={report.NegativeWeights:N0} zero={report.ZeroWeights:N0} pos={report.PositiveWeights:N0})");
Console.WriteLine($"      done in {(sw.Elapsed - t0).TotalSeconds:F1}s");

// 5. Pack + save.
t0 = sw.Elapsed;
Console.WriteLine($"[5/5] Packing + saving {output}...");
var flat = FlatParameterPack.Pack(transformer);

var vocabList = new string[tokenizer.VocabSize];
for (var id = 0; id < tokenizer.VocabSize; id++)
{
    vocabList[id] = tokenizer.GetTokenString(id);
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
    vocabulary: vocabList,
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
