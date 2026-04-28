# TruckMate corpus + small-preset ceiling analysis

Companion to [`scaling-truckmate-corpus-v1.0.md`](scaling-truckmate-corpus-v1.0.md).
That doc covers v1 -> v2 corpus mechanics. This doc answers the next-tier
question: **how big can the corpus get before we stop seeing useful
intent-classification gains, and how big can the model get before phone
deployment becomes infeasible?**

## TL;DR

| Axis | Headroom | Bottleneck after that |
|---|---|---|
| Corpus expansion | ~10x more cities/templates beyond v2 (~500K examples) | `WordLevelTokenizer` 5174 vocab cap; needs BPE rewrite |
| Model size for phone | small (~7M params) is the only interactive option | medium/large tank to <2 tok/sec on Cortex-X4 mid-tier silicon |
| Per-domain quality | v1 (50K) already past task-saturation for the 16 intent families | adding intent families, not examples, is the next gain |

## Corpus expansion ceiling

`WordLevelTokenizer` is hard-capped at 5174 distinct tokens per the
v1 -> v2 design (the cap freezes the flat-parameter length so previously
serialized weights stay shape-compatible). Practical growth budget:

| Source | Tokens used | Tokens remaining (of 5174) |
|---|---|---|
| Special tokens (`<bos>`, `<eos>`, `<unk>`) | 3 | 5171 |
| v1 corpus (50K examples, seed=42) | ~3500 | ~1671 |
| v2 corpus extras (cities + weather + time-of-day, +50/+15/+10) | ~150 | ~1521 |
| Headroom for hypothetical v3 | ~1500 (~ten more US-region splits) | ~0 |

Hard ceiling: **roughly 500K examples / 100-150 cities / 15-20 chains /
a couple dozen weather + time-of-day pools** before vocab saturates and
the BPE-tokenizer rewrite becomes mandatory.

Empirically, TruckMate's *task* is bounded by intent-family count, not
example count. With 16 families (`start_trip`, `stop_trip`, `navigate`,
`find_poi`, `route_preference`, `hos_status`, `hos_break_check`,
`hos_drive_remaining`, `add_todo`, `add_expense`, `eta_query`,
`next_stop_query`, `trip_status`, `reroute`, `update_load`, plus v2 split
variants), most intent classifiers plateau between **10K and 100K
diverse examples**. Doubling the corpus past 200K should produce
sub-5% intent-accuracy improvements unless new families are introduced.

**Practical recommendation for the 10-day window:**
- Stay on v1 (50K) for the first phone-deployable build.
- v2 (200K) is worth building only when adding the ASR-noise pool or
  the multi-turn dialogue corpus (both require new generators).
- Anything past v2 should be paired with the BPE upgrade.

## Model size ceiling on phone

Reference target: Motorola Edge 2024 (Snapdragon 8s Gen 3, Cortex-X4,
8 GB RAM, Mono Android, no `AdvSimd` intrinsics, only the portable
`Vector<float>.IsHardwareAccelerated` path).

| Preset | Params | Token-emb size | Resident bytes (est) | Per-decode-token (est) | Phone interactive? |
|---|---|---|---|---|---|
| small | ~7M | vocab x 256 | ~22 MiB | 70-100 ms | yes (~10-14 tok/sec) |
| medium | ~56M | vocab x 512 | ~180 MiB | 0.5-1.5 sec | borderline |
| large | ~121M | vocab x 768 | ~400 MiB | 1.5-3 sec | no |

Estimates derive from Bonsai 782M's measured 7-8 sec/decode-token on the
same phone, scaled by the `EstimateParams` ratio (linear approximation
of CPU-bound work, validated against the full Bonsai number after
KV-FU8 + KV-FU6 NEON kernels).

Three knobs that shift the curve:
1. **CoreCLR-on-Android** (opt-in flag) unlocks `AdvSimd` intrinsics
   and the hand-rolled NEON DotInt8 path. Estimated 2-3x speedup;
   medium becomes borderline-interactive.
2. **Dual-int8 SDOT** (KV-FU9 candidate) requires NEON intrinsics
   that the portable `Vector<float>` path can't emit. Another ~1.5x
   on top of CoreCLR. Still doesn't make large interactive.
3. **NPU offload** (Hexagon DSP via Qualcomm AI Engine SDK).
   Architecturally separate workstream; 10-50x potential but
   weeks-of-effort price tag.

## Chinchilla-vs-task-saturation gap

For raw next-token prediction loss, the [Chinchilla](https://arxiv.org/abs/2203.15556)
optimum is ~20 training tokens per parameter:

| Preset | Chinchilla token budget | v1 corpus tokens (50K x 128) | v2 corpus tokens (200K x 128) |
|---|---|---|---|
| small (7M) | 140M | 6.4M (severely under-trained) | 25.6M (under-trained) |
| medium (56M) | 1.1B | 6.4M (negligible) | 25.6M (severely under-trained) |
| large (121M) | 2.4B | 6.4M | 25.6M |

But: TruckMate's task (16 intent families, narrow slot vocabulary,
deterministic JSON output format) has a far lower information capacity
than open-domain language modeling. Empirically:

- Intent-classification accuracy plateaus around **20-50K diverse
  examples** for an SLM in the 5-100M parameter range.
- Once the model has memorized the 16 intent-family templates and
  the slot fillers, additional examples add noise, not signal.
- Loss curves keep declining (perplexity over the entire continuation
  improves) but the *intent-name argmax* and *first-N-token JSON-shape*
  metrics flatten.

This means the small preset is not as under-trained as the Chinchilla
table suggests for this specific task. The IntentBench page in the
MAUI app measures the metric that matters: `expected_intent_name`
appearing in the first 24 decoded tokens.

## What to build next

Concrete corpus + model bets after the 10-day deployment:

1. **Multi-turn dialogue corpus** (`MultiTurnCorpusGenerator` already
   stubbed in `Distributed.Coordinator/Services/`). Requires reasoning
   about turn-context budget vs the small preset's 128-token sequence
   length. Likely caps at ~3-turn conversations on the small preset.
2. **Real ASR-trace ingestion** (`SonnetAsrCorpusGenerator` already
   present). PII-scrub workstream is the blocker, not modeling.
3. **CoreCLR-on-Android opt-in** (Android workload pivot). Unlocks
   `AdvSimd` -> medium becomes interactive on the phone.
4. **BPE tokenizer** (~500 LOC swap). Required before the corpus passes
   ~500K diverse examples or covers Canadian/European geography.

Until those land, the right product target is:
- **small preset (~7M params)** trained on **v1 (50K examples)** for
  **1-3 epochs**, deployed via MAUI MauiAsset extract path.
- IntentBench-measured intent-substring accuracy as the phone-side gate.
- Bonsai retired as a phone target (research curiosity only).
