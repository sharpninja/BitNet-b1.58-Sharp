# TruckMate distributed-training runbook

Step-by-step to spread training across PAYTON-DESKTOP (coordinator + worker)
and PAYTON-LEGION2 (worker), then export the trained checkpoint to
`truckmate-small.tmv1` for the MAUI app + the TruckMate repo's hosted-agent.

The pipeline uses the existing `BitNetSharp.Distributed.Coordinator` host
+ `BitNetSharp.Distributed.Worker` daemons. `WordLevelInferenceModel` on
the inference end consumes the same vocab the coordinator's
`tokenize-corpus` produced, so checkpoints are interchangeable between
single-machine `tools/TruckMateTrainer` and the distributed pipeline.

## 0. Prereqs

- Both machines: .NET 10 SDK installed, repo cloned at the same path.
- PAYTON-DESKTOP reachable from PAYTON-LEGION2 over the local LAN. Pick a
  shared port (default coordinator listens on `5001` or whatever
  `Coordinator:BaseUrl` configures).
- Shared API key. Generate one and set on both machines:
  ```
  $env:Coordinator__WorkerApiKey = "<random-32+ char string>"
  ```
  Workers send it as `X-Api-Key`; coordinator validates.

## 1. PAYTON-DESKTOP - generate corpus + tokenize + start coordinator

```powershell
# From repo root on PAYTON-DESKTOP.
$DataRoot = "F:\ProgramData\BitNetCoordinator"
$env:Coordinator__DatabasePath = "$DataRoot\coordinator.db"
$env:Coordinator__WorkerApiKey = "<shared-key>"
$env:Coordinator__ModelPreset  = "small"   # truckmate-small (~7M params)

# Generate full v1 corpus (50K examples) into the coordinator's data dir.
dotnet run -c Release --project src\BitNetSharp.Distributed.Coordinator -- `
    generate-corpus 50000 --seed 42 --pool v1 --name truckmate-v1

# Train the WordLevelTokenizer over the corpus + write 10 binary shards.
# Cap at 5174 vocab (preserves weight-shape compatibility with v2 / v3).
dotnet run -c Release --project src\BitNetSharp.Distributed.Coordinator -- `
    tokenize-corpus 5174 truckmate-v1

# Seed N training tasks of `tokensPerTask` each. Each task is a window
# of pre-tokenized shard bytes the worker downloads + trains on.
dotnet run -c Release --project src\BitNetSharp.Distributed.Coordinator -- `
    seed-real-tasks 200 262144

# Start the coordinator host.
dotnet run -c Release --project src\BitNetSharp.Distributed.Coordinator
```

Expected: coordinator boots, admin UI live at `https://localhost:5001/admin/dashboard`,
worker plane accepts X-Api-Key requests at the same host. Verify by hitting
`/admin/training-status` from a browser - shows live worker registrations
+ task counts as workers come online.

## 2. PAYTON-DESKTOP - start a local worker

```powershell
$env:Worker__CoordinatorBaseUrl = "https://localhost:5001"
$env:Worker__ApiKey             = "<shared-key>"
$env:Worker__WorkerId           = "payton-desktop"

dotnet run -c Release --project src\BitNetSharp.Distributed.Worker
```

The worker registers, runs a startup capability calibration (~30 s of
real training-step throughput), then claims tasks until the queue
drains.

## 3. PAYTON-LEGION2 - start a remote worker

```powershell
$env:Worker__CoordinatorBaseUrl = "http://PAYTON-DESKTOP:5001"
$env:Worker__ApiKey             = "<shared-key>"
$env:Worker__WorkerId           = "payton-legion2"

dotnet run -c Release --project src\BitNetSharp.Distributed.Worker
```

Optional: spin up a second worker on LEGION2 with `WorkerId =
"payton-legion2-2"` if the box has multiple cores worth dedicating.

## 4. Watch progress

Open the admin dashboard at `https://PAYTON-DESKTOP:5001/admin/dashboard`
or the live training-status page. Both show:
- Per-worker tasks claimed / completed / mean-throughput
- Global weight version increment rate
- Per-corpus rollup rows for `truckmate-v1-` shard prefix

Rough numbers for the small preset (7M params, 5174 vocab):
- One worker on a Ryzen-class desktop sustains ~500 tokens/sec real
  training throughput. At 6.4M tokens (50K x 128) per epoch, that's
  ~3.5 hr / epoch / worker.
- Two workers cut that to ~1.75 hr / epoch.
- 3 epochs across both machines: **~5-6 hr wall-clock total**.

## 5. Export the trained checkpoint

When the queue drains and the global weight version stops climbing
(or hits a target version), export the latest weights to a `.tmv1`
file:

```powershell
# On PAYTON-DESKTOP (where the weight store lives):
dotnet run -c Release --project src\BitNetSharp.Distributed.Coordinator -- `
    export-tmv1 F:\GitHub\BitNet-b1.58-Sharp\data\truckmate\truckmate-small.tmv1
```

Output: `truckmate-small.tmv1` (~26 MiB at small-preset shape) holding
the BitNetConfig fields, the WordLevelTokenizer's id-ordered vocab, and
the FlatParameterPack float[].

Pass `--version <N>` to export an earlier version instead of the latest:

```powershell
dotnet run -c Release --project src\BitNetSharp.Distributed.Coordinator -- `
    export-tmv1 my-snapshot.tmv1 --version 42
```

## 6. Validate the checkpoint

```powershell
# X86 dev-box smoke test:
dotnet run -c Release --project tools\TruckMateValidator -- `
    F:\GitHub\BitNet-b1.58-Sharp\data\truckmate\truckmate-small.tmv1
```

Reports per-prompt TTFT, per-decode-token p50/p95, intent-substring
accuracy. If accuracy is north of 70% on the 8 validator prompts, the
checkpoint is ready for phone deployment.

## 7. Ship to phone

Rebuild + install the MAUI APK:

```powershell
dotnet build benchmarks\BitNetSharp.Benchmarks.Maui -c Release -f net10.0-android
adb install -r benchmarks\BitNetSharp.Benchmarks.Maui\bin\Release\net10.0-android\android-arm64\publish\com.companyname.bitnetsharp.benchmarks.maui-Signed.apk
```

Open the app, hit Chat or Intent tab, run a prompt. The MAUI loader
reads `truckmate-small.tmv1` from the APK assets, builds a
`WordLevelInferenceModel` over the same vocab the coordinator trained
against, and streams responses with per-token decode latency surfaced
to the UI.

## 8. TruckMate-app integration

The TruckMate repo can consume the same `truckmate-small.tmv1` via:

```csharp
var (header, flat) = TruckMateModelStore.Load(path);
var cfg = new BitNetConfig(
    vocabSize: header.VocabSize,
    dimension: header.Dimension,
    /* ... */
    kvHeadCount: header.KvHeadCount);
var transformer = new BitNetTransformer(cfg, NullLogger<BitNetTransformer>.Instance,
    seed: 42, skipRandomInit: true);
FlatParameterPack.Unpack(transformer, flat);

// Build tokenizer from header.Vocabulary (round-trip via temp JSON file
// since WordLevelTokenizer's only public ctor is LoadFromFile).
var tokenizer = LoadVocabFromArray(header.Vocabulary);

var model = new WordLevelInferenceModel(transformer, tokenizer);
await foreach (var token in model.StreamGenerateAsync(prompt))
{
    // hand to Microsoft Agent Framework hosted-agent
}
```

`WordLevelInferenceModel` is a regular `sealed class` in
`BitNetSharp.Core`, no host dependencies. Reference Core +
Distributed.Contracts and you have a self-contained inference path.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| Worker says `401 Unauthorized` | API key mismatch | Confirm `Coordinator__WorkerApiKey` matches across both machines |
| `export-tmv1` says vocab size differs from preset | Coordinator's preset changed mid-train | Set `Coordinator__ModelPreset` consistently OR re-tokenize |
| Validator reports `vocab_size mismatch` | TMV1 saved with WordLevel vocab but loaded against BitNet model | Always use `WordLevelInferenceModel` for coordinator-exported checkpoints (current default) |
| Worker exits with `flat length N != expected M` | `Coordinator:ModelPreset` and existing weight version disagree | Delete `weights/v*.bin` to reset, or set preset to match |
| `tokenize-corpus` exit code 3 | Vocab passed 5174 cap | Reduce v2 expansion or upgrade to BPE (out of scope for current PR) |
