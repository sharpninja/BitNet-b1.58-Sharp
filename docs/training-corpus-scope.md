# Training Corpus Scope and Sources

## Purpose

Defines the scope, content, and generation pipeline of the training
corpus used by the BitNet b1.58 Sharp distributed training fleet.
This is not a general-purpose language corpus. It is a
**domain-specific intent-classification corpus** for the TruckMate
voice-assistant small language model (SLM).

## Scope boundaries

**In scope:**

- Short, single-turn `[USER] <utterance> [INTENT] <json>` pairs.
- Trucker voice commands mimicking noisy ASR input (CB shorthand,
  filler words like "uh"/"um", geographic aliases).
- Ten intent families: trip management, navigation, find-POI, route
  preferences, HOS/compliance, to-do items, expenses, status queries,
  rerouting, and load updates.
- US-only geography (50 major cities, 20 interstates, 10 truck-stop
  chains).
- American English, lowercase-normalized.

**Out of scope:**

- Multi-turn dialogue / conversation history.
- General chat, open-domain Q&A, reasoning tasks.
- Non-English utterances.
- PII, real driver data, real dispatch logs, real telematics.
- Any licensed third-party text.

**Why domain-specific:** the deployed SLM serves a narrow
classification surface. Keeping the corpus inside that surface lets a
~56M-parameter `truckmate-medium` preset fit the task cleanly without
the data/compute budget a general LM would need. The trade-off is
documented: scaling to the `truckmate-large` (~121M) preset requires
growing the corpus to 200K+ examples to avoid overfitting — see
`src/BitNetSharp.Distributed.Contracts/TruckMateModelPresets.cs`.

## Source

**Single source: synthetic generation.** No external datasets, no
scraped web text, no licensed corpora. Every row comes from
`TruckMateCorpusGenerator.Generate` at
`src/BitNetSharp.Distributed.Coordinator/Services/TruckMateCorpusGenerator.cs`.
Generation is deterministic under `Random(42)` so corpus runs are
reproducible.

Template lists the generator combines:

| Category | Count | Examples |
|---|---|---|
| US cities | 50 | Dallas, Memphis, Salt Lake City, … |
| Truck-stop chains | 10 | Flying J, Pilot, Love's, Buc-ee's, … |
| Interstates | 20 | I-10, I-35, I-95, … |
| Stop types | 12 | truck stop, fuel, scale, tire shop, … |
| Route preferences | 12 | avoid tolls, shortest route, … |
| ASR noise fillers | 9 | "uh ", "um ", "like ", "" (70% silent) |

## Generation pipeline

Two CLI subcommands on the coordinator DLL, both running out of
`src/BitNetSharp.Distributed.Coordinator/Program.cs`:

1. **`generate-corpus [count]`** — default `count=50000`. Writes
   `corpus/truckmate-shard-NNNN.txt` (5,000 examples per shard, so
   default = 10 shards) plus `corpus/manifest.json`. Each line is one
   training example in the format:
   ```
   [USER] take me to the flying j in dallas [INTENT] {"intent":"navigate","slots":{"destination":"Flying J","city":"Dallas"}}
   ```

2. **`tokenize-corpus [maxVocab]`** — default `maxVocab=8000`. Trains
   a `WordLevelTokenizer` across every `*.txt` shard, writes
   `corpus/tokenized/vocab.json` plus one
   `corpus/tokenized/truckmate-shard-NNNN.bin` per shard (packed
   little-endian int32 token ids).

The split is deliberate: text shards are human-readable for audit,
binary shards are what workers stream during training. Workers never
see plaintext — the coordinator hands out shard byte ranges and
workers read int32 ids directly.

## Current corpus (`truckmate-v1`)

| Property | Value |
|---|---|
| Manifest name | `truckmate-v1` |
| Total examples | 50,000 |
| Shards | 10 (`truckmate-shard-0000` … `truckmate-shard-0009`) |
| Examples per shard | 5,000 |
| Vocabulary size | 5,174 tokens |
| Tokens per shard | ~183,000 int32 |
| Tokens total | ~1.83M |
| RNG seed | 42 (deterministic) |

Vocab size is lower than the `maxVocab=8000` target because the
domain vocabulary saturates — the template vocabulary collapses to
~5K unique word-level tokens across 50K examples.

## How the distributed trainer consumes it

Tasks are enqueued via `seed-real-tasks [tokensPerTask] [maxTasksPerShard]`.
Each task holds a byte range `(shardId, offset, length)` pointing
into a `.bin` file. `CorpusShardLocator.Resolve(shardId)` walks:

1. `corpus/tokenized/{id}.bin` (preferred — int32 token stream)
2. `corpus/{id}.bin`
3. `corpus/{id}.txt`
4. bare id (for legacy / synthetic fixtures)

Workers download the slice via `GET /corpus/{shardId}?offset=X&length=Y`,
reinterpret the bytes as `int32[]`, and feed the token stream into
the forward/backward pass.

## Retraining the corpus

```
dotnet run --project src/BitNetSharp.Distributed.Coordinator \
    -- generate-corpus 50000
dotnet run --project src/BitNetSharp.Distributed.Coordinator \
    -- tokenize-corpus 8000
```

Both commands are idempotent and overwrite existing shards. Bumping
`count` grows shards proportionally. The vocab is retrained every
run, so downstream worker weights become incompatible with the new
vocab — trigger a global weight reset if the vocab size changes.

## Upgrade paths

- **Corpus-v2 (larger)** — scale to 200K+ examples to unlock the
  `truckmate-large` preset. Add more cities / stop types / variants
  to avoid template memorization.
- **Corpus-v2 (multi-turn)** — add optional prior-turn context. Out
  of scope for current pipeline; would require a new generator.
- **Corpus-v3 (real ASR captures)** — replace synthetic `Noise()`
  with real truck-cab ASR traces. Requires a data-collection
  workstream and a PII-scrubbing step that does not exist today.

All three are deferred. Current priority is finishing distributed
training on `truckmate-v1` before growing the corpus surface.
