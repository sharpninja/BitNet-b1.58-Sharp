# Scaling TruckMate Corpus v1.0 → v2 (200K examples)

Plan for scaling the synthetic training corpus from `truckmate-v1`
(50,000 examples, 10 shards) to `truckmate-v2` (200,000 examples, 20
shards). Unlocks the `truckmate-large` (~121M-param) preset without
overfitting the smaller v1 surface. Preserves determinism, pins the
vocab at 5174 so existing weights stay shape-compatible, and lets v1
and v2 coexist on disk and in the work queue.

## Context

- **v1 state.** `TruckMateCorpusGenerator` currently produces 50K
  examples seeded with `Random(42)`, packaged into 10 text shards
  plus 10 binary shards at vocab 5174. `truckmate-large` in
  `src/BitNetSharp.Distributed.Contracts/TruckMateModelPresets.cs`
  is documented as "requires 200K+ examples to avoid overfitting."
- **Tokenizer.** Custom `WordLevelTokenizer` in
  `src/BitNetSharp.Distributed.Contracts/WordLevelTokenizer.cs`
  (word-level, not BPE / not TinyLlama). Honors `maxVocab` on train.
- **Shard resolution.** `CorpusShardLocator.TryResolve` walks
  `corpus/tokenized/{id}.bin` → `corpus/{id}.bin` →
  `corpus/{id}.txt` → bare id. Any `shardId` prefix works.
- **Wire contract.** `WorkTaskAssignment` carries only `ShardId` —
  no corpus-version field. The shard-ID prefix
  (`truckmate-v2-shard-NNNN`) carries the version implicitly.
- **Preset math.** `EstimateParams` is linear in vocab size. If
  vocab grows, flat-param length changes and the coordinator's
  weight-version hard-reset fires. Vocab must stay at 5174.

## Design decisions

- **D1.** v2 coexists with v1 on disk. Shard IDs:
  `truckmate-v2-shard-0000` … `truckmate-v2-shard-0019`.
- **D2.** 200,000 examples, 20 shards × 10,000 each.
- **D3.** Vocab pinned at 5174. Enforced by default `maxVocab=5174`
  in `tokenize-corpus` plus a hard assertion (exit 3) if the trained
  vocab exceeds 5174.
- **D4.** Generator gains explicit `seed` (default 42) and
  `poolVersion` enum (`V1` frozen, `V2` expanded). The canonical v2
  corpus is defined by `seed=42 + poolVersion=V2 + manifestName=truckmate-v2`.
- **D5.** No wire-contract change; version encoded only in the
  shard-ID prefix.
- **D6.** No preset value changes; doc-comments updated.
- **D7.** Canadian geography deferred to a future v3 to stay inside
  the 5174 vocab budget.
- **D8.** `weather_alert` folds into `reroute` as a variant and
  `multi_stop_trip` folds into `start_trip` as a variant. The
  intent-family switch stays at `rng.Next(10)` so v1 analytics
  distributions are preserved.

## Files changed

| File | Change |
|---|---|
| `src/BitNetSharp.Distributed.Coordinator/Services/TruckMateCorpusGenerator.cs` | `CorpusPoolVersion` enum; extended `Generate` signature (`seed`, `poolVersion`, `manifestName`, `examplesPerShard`); nested `TruckMatePools` class keyed by version; V2 branches for reroute (25% weather variant) / start_trip (20% multi-stop) / navigate (30% time-of-day slot). |
| `src/BitNetSharp.Distributed.Coordinator/Program.cs` | `generate-corpus` parses `--seed`, `--pool v1\|v2`, `--name`, `--examples-per-shard`. `tokenize-corpus` default `maxVocab` 8000 → 5174; optional shard-prefix positional; backs up `vocab.json` → `vocab.v1.json`; exits 3 if vocab > 5174. |
| `src/BitNetSharp.Distributed.Contracts/TruckMateModelPresets.cs` | Class-level + `Large` preset doc-comments updated to reference v1/v2 coexistence and 5174 vocab pinning. No code change. |
| `tests/BitNetSharp.Tests/TruckMateCorpusGeneratorTests.cs` (new) | 7 xUnit tests. See Phase 5. |
| `scripts/Generate-TruckMateCorpusV2.ps1` (new) | PowerShell rollout wrapper, parameterized; no hard-coded paths. |

## Phased implementation

### Phase 1 — Generator refactor (non-breaking defaults)

1. `public enum CorpusPoolVersion { V1, V2 }`.
2. Extended signature:
   ```csharp
   public static CorpusManifest Generate(
       string outputDirectory,
       int count = 50_000,
       int examplesPerShard = 5_000,
       int seed = 42,
       CorpusPoolVersion poolVersion = CorpusPoolVersion.V1,
       string manifestName = "truckmate-v1")
   ```
   Defaults reproduce the v1 byte stream, so callers that have not
   been updated keep working.
3. `new Random(42)` → `new Random(seed)`.
4. Pool arrays moved into `internal static class TruckMatePools`
   with `GetCities`, `GetTruckStopChains`, `GetInterstates`,
   `GetStopTypes`, `GetRoutePrefs`, `GetAsrNoise`, `GetWeather`,
   `GetTimeOfDay` accessors. V1 getters return the frozen arrays;
   V2 getters return cached concatenations where V2 is a strict
   superset of V1 (cities, chains, ASR noise) or V2-only pools
   (weather, time-of-day).
5. `poolVersion` threaded through every `Generate*Command`.
6. V2 variant branches:
   - `GenerateTripCommand`: 20% of start-trip rolls emit multi-stop
     variant with slot `"stops":["{c1}","{c2}"]`.
   - `GenerateRerouteCommand`: 25% of reroute rolls emit weather-
     alert variant with slots `"weather":"{w}","via":"{interstate}"`.
   - `GenerateNavigateCommand`: 30% of navigate utterances inject
     `"when":"{time}"` into slots.
7. Shard ID format: `$"{manifestName}-shard-{shardIndex:D4}"`.
8. Manifest written as `manifest.{manifestName}.json` so v1 and v2
   manifests coexist. A legacy `manifest.json` is still written for
   v1 to preserve back-compat with readers that look for it.
9. `CorpusManifest` record gains `Seed` (int) + `PoolVersion` (string).

### Phase 2 — V2 template pools (actual content)

Added to `TruckMatePools.V2` as a superset of V1:

**V2 city extensions (+50 → 100 total US freight hubs):**
Toledo, Reno, Spokane, Tucson, Fort Worth, Knoxville, Chattanooga,
Mobile, Greensboro, Lexington, Fresno, Bakersfield, Scranton,
Harrisburg, Roanoke, Macon, Montgomery, Gulfport, Beaumont, Lubbock,
Amarillo, Wichita, Topeka, Sioux Falls, Fargo, Rapid City, Billings,
Cheyenne, Grand Junction, Flagstaff, Medford, Eugene, Yakima, Twin
Falls, Pocatello, Laramie, Casper, Bismarck, Duluth, Eau Claire,
Green Bay, Dubuque, Davenport, Springfield, Bloomington, Evansville,
Bowling Green, Asheville, Wilmington, Erie.

**V2 truck-stop-chain extensions (+10 → 20 total):**
Stuckey's, Iowa 80, Roady's, AmBest, Pilot Travel, FleetPride,
AllStar Travel, Rip Griffin, Little America, Travel America.

**V2 weather pool (NEW, 15):** rain, heavy rain, snow, ice, fog,
high winds, thunderstorm, freezing rain, blizzard, whiteout, dust
storm, wildfire smoke, flooding, black ice, sleet. Used only by the
V2 reroute weather-alert branch.

**V2 time-of-day pool (NEW, 10):** this morning, tonight, by
sunrise, before dark, late, first thing tomorrow, this afternoon,
after my break, by end of shift, at dawn. Injected into trip /
navigate utterances at 30% probability when V2.

**V2 ASR-noise extensions (+6 → 15):** "so ", "well ", "okay ",
"kinda ", "basically ", "I mean " concatenated with existing V1
fillers.

**Multi-stop start_trip variant:** utterance templates
`"pick up in {c1} then drop in {c2}"` and variants; slot
`"stops":["{c1}","{c2}"]`.

**weather_alert reroute variant:** utterance templates
`"reroute around {w}"` and `"there's {w} on {interstate}"`; slots
`"weather":"{w}"` and `"via":"{interstate}"`.

Vocabulary budget: v1 trains ~3.5K unique words out of a 5174
target. New template tokens add roughly 150 unique words (city names
dominate). Headroom ≈ 1.5K; the tripwire in Phase 3 catches any
overflow.

### Phase 3 — CLI wiring (Program.cs)

`GenerateCorpusCommandLine` accepts optional flags after the
positional `count`:

- `--seed N` (default 42)
- `--pool v1|v2` (default v1)
- `--name <manifestName>` (default `truckmate-v1`)
- `--examples-per-shard N` (default 5000)

Canonical v2 invocation:
```
coordinator generate-corpus 200000 --seed 42 --pool v2 --name truckmate-v2 --examples-per-shard 10000
```

`TokenizeCorpusCommandLine`:

- Default `maxVocab` 8000 → 5174.
- Optional 2nd positional: `shardPrefix`. When set, glob is
  `{prefix}-*.txt` (not `*.txt`).
- Before overwriting `vocab.json`, if a shard-prefix is set and
  `vocab.v1.json` does not already exist, copies existing
  `vocab.json` → `vocab.v1.json`.
- After `WordLevelTokenizer.TrainFromCorpus`, hard-fails with exit
  code 3 if `tokenizer.VocabSize > 5174`.

### Phase 4 — Preset doc-comment updates

Text-only changes to `TruckMateModelPresets.cs`:

- Class-level `<para>`: "Vocab pinned at 5174 across corpus
  versions" — explains v1/v2 weight compatibility.
- `Large` preset `<summary>`: "Trained on truckmate-v2 (200K
  examples, vocab-compatible with truckmate-v1)."

No code change, no `EstimateParams` change, no preset value change.
FlatParameterPack length identical across corpus versions.

### Phase 5 — Tests

`tests/BitNetSharp.Tests/TruckMateCorpusGeneratorTests.cs` (gated
`#if NET10_0_OR_GREATER` because the coordinator project is only
referenced on the net10 slice). Seven tests:

1. `Generate_seed42_V1_is_byte_deterministic` — regression guard
   for v1 byte parity after refactor.
2. `Generate_seed42_V2_is_byte_deterministic` — same for v2.
3. `Generate_V1_and_V2_produce_different_byte_streams` — sanity
   check pools actually differ.
4. `Generate_V2_utterance_collisions_under_threshold` — 10K v2
   draws; asserts unique-utterance ratio > 0.30. Threshold set
   empirically — finite templates collide at 10K draws.
5. `Generate_V2_intent_distribution_roughly_matches_V1` — 10K of
   each; per-family drift < 30% for families with ≥ 50 v1 examples.
   Confirms variant folding preserves family distribution.
6. `Tokenize_V2_sample_stays_under_5174_vocab_cap` — generates 5K
   v2 examples, trains the tokenizer in-process, asserts
   `VocabSize <= 5174`. Replaces the earlier plan's
   `TokenizeCorpusCommandLine_ForTest` hook idea: calling the
   tokenizer directly exercises the same logic without needing an
   internal entry point or child process.
7. `Generate_writes_versioned_manifest_file` — asserts
   `manifest.truckmate-v2.json` lands in the output dir.

### Phase 6 — Rollout

Coexists with in-flight training; no disruption to v1 tasks mid-flight.

1. Merge Phases 1–5 locally; build green; tests green.
2. Commit + push to `azure`, mirror to `origin`.
3. Redeploy coordinator DLL via
   `scripts/deploy-coord.ps1` (robocopy + service restart).
4. Run `scripts/Generate-TruckMateCorpusV2.ps1 -Coordinator <host>
   -RepoRoot <path> [-DataRoot <path>]`. The wrapper:
   - Deploys the DLL (unless `-SkipDeploy`).
   - Invokes `generate-corpus 200000 --seed 42 --pool v2
     --name truckmate-v2`.
   - Backs up `vocab.json` → `vocab.v1.json` if not already backed up.
   - Invokes `tokenize-corpus 5174 truckmate-v2`.
   - Aborts and restores the v1 vocab if the tokenizer exits 3.
5. Manually run `seed-real-tasks` against v2 shards (the script
   emits a warning here because `seed-real-tasks` does not yet
   honor a `--shard-prefix` flag — a follow-up task).
6. Monitor `/admin/dashboard` for loss regression; first-epoch v2
   loss should be within 10% of v1's first-epoch loss on the same
   preset.
7. Rollback: restore `vocab.v1.json` → `vocab.json`, re-enqueue
   against v1 shard IDs. v1 `.bin` files are untouched throughout.

## Verification

1. `dotnet test BitNet-b1.58-Sharp.slnx --framework net10.0 --filter FullyQualifiedName~TruckMateCorpusGeneratorTests`
   — 7/7 green.
2. `dotnet build BitNet-b1.58-Sharp.slnx -c Release` — 0 warnings,
   0 errors.
3. Remote smoke: `scripts/Generate-TruckMateCorpusV2.ps1` runs to
   completion; `Invoke-WebRequest http://<host>:5000/corpus/truckmate-v2-shard-0000?offset=0&length=1024`
   returns 200 with 1024 bytes.
4. End-to-end: seed ≥ 40 v2 tasks, watch `/admin/dashboard` Done
   count rise over 30 minutes with no claim-expiry loops. (The
   measured-throughput lease calibration from the P0 commit is
   already deployed.)

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| V2 pool expansion pushes vocab > 5174 | Medium | CLI tripwire (exit 3) + Phase 5 test 6 |
| Refactor breaks V1 byte parity | High (if refactor rushed) | Phase 5 test 1 |
| vocab.json overwrite corrupts in-flight v1 training | Medium | `Generate-TruckMateCorpusV2.ps1` backs up before overwrite |
| Seed=42 RNG state advances differently V1 vs V2 | Low | Accepted — v2 is a new corpus, not a v1 extension |
| Worker LRU thrashes across 20 shards | Low | Worker-local sharding keeps affinity |
| Reroute / start_trip variant folding shifts per-family counts | Low | Phase 5 test 5 gates this |

## Out of scope (deferred)

- Canadian geography → v3.
- Multi-turn dialogue corpus → v3 (new generator needed).
- Real ASR-trace ingestion → needs PII-scrubbing workstream.
- `seed-real-tasks --shard-prefix` flag → tracked as P1 follow-up
  in `docs/state-of-completion.md`.
