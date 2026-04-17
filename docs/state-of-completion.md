# State of Completion and Remaining Work

Snapshot of project completion status. Updated 2026-04-17.

This is the "where we are right now" doc: what works end-to-end, what
is partially wired, and what is still on the backlog. Cross-references
the detailed plans in `distributed-training.md`,
`training-corpus-scope.md`, and
`full-implementation-plan-real-training-benchmarks-purity-v1.0.md`.

## Status legend

- ✅ **Done** — shipped, exercised in production/dev, no known gaps.
- 🟡 **Partial** — primary path works; edge cases or follow-ups open.
- 🔴 **Blocked / bug** — known defect on an otherwise-delivered feature.
- ⚪ **Not started** — on the backlog, no code yet.

## 1. Core model (`BitNetSharp.Core`)

| Component | Status | Notes |
|---|---|---|
| BitLinear ternary projection | ✅ | Paper-aligned, bit-packed storage |
| RmsNorm | ✅ | Fused with BitLinear residual |
| Rotary position embedding (RoPE) | ✅ | Head-dim must be even — asserted |
| Causal multi-head attention | ✅ | |
| SwiGLU feed-forward | ✅ | |
| `BitNetTransformer` end-to-end | ✅ | Forward + backward verified |
| `FlatParameterPack` serialize/load | ✅ | Used by distributed fleet |
| `TruckMateModelPresets` (small/medium/large) | ✅ | Invariants asserted |

## 2. Hosting & CLI (`BitNetSharp.App`)

| Feature | Status | Notes |
|---|---|---|
| `chat` subcommand | ✅ | MAF-oriented host |
| `datagen` subcommand | ✅ | Seed-file driven JSON synthesis |
| `visualize` subcommand | ✅ | Weight summary + inspection |
| BenchmarkDotNet model comparison | ✅ | Local-only, docs in `benchmarking.md` |

## 3. Corpus pipeline

| Stage | Status | Notes |
|---|---|---|
| `generate-corpus` CLI | ✅ | `TruckMateCorpusGenerator`, deterministic seed=42 |
| v1 text shards (`corpus/truckmate-v1-*.txt`) | ✅ | 10 shards × 5K examples |
| **v2 text shards (`corpus/truckmate-v2-*.txt`)** | ✅ | **20 shards × 10K examples, deployed 2026-04-17** |
| `tokenize-corpus` CLI | ✅ | `WordLevelTokenizer`, vocab pinned at 5,174 |
| v1 binary shards (`corpus/tokenized/truckmate-v1-*.bin`) | ✅ | int32 LE, ~1.83M tokens total |
| **v2 binary shards (`corpus/tokenized/truckmate-v2-*.bin`)** | ✅ | **int32 LE, 7.47M tokens total (~4× v1)** |
| Corpus manifests | ✅ | `manifest.truckmate-v1.json`, `manifest.truckmate-v2.json` |
| `vocab.v1.json` backup | ✅ | Preserved; rollback path intact |
| Corpus scope doc | ✅ | `docs/training-corpus-scope.md` |
| Scaling plan v1.0 doc | ✅ | `docs/scaling-truckmate-corpus-v1.0.md` |
| Real-ASR trace ingestion | ⚪ | Deferred — requires PII scrubbing workstream |
| Multi-turn corpus | ⚪ | Deferred — new generator needed |

## 4. Coordinator

| Feature | Status | Notes |
|---|---|---|
| ASP.NET host + Windows Service | ✅ | Installed on PAYTON-DESKTOP:5000 |
| OIDC admin auth (Duende IdentityServer) | ✅ | Seeded TestUser from options |
| Shared `X-Api-Key` worker auth | ✅ | Supersedes per-client OAuth |
| `SqliteWorkQueueStore` | ✅ | Dequeue + lease + status transitions |
| `SqliteWorkerRegistryStore` | ✅ | Heartbeat + calibration telemetry |
| `SqliteTelemetryStore` | ✅ | gradient_events + per-worker measured tps |
| `SqliteLogStore` | ✅ | Worker log ring, filter API |
| `FileSystemWeightStore` | ✅ | Versioned flat blob |
| Weight-version hard-reset on shape mismatch | ✅ | Banner + `latestVersion + 1` |
| `/corpus/{shardId}` route | ✅ | Serves .bin or .txt |
| `CorpusShardLocator` resolution order | ✅ | tokenized → .bin → .txt → bare |
| `GET /work` claim endpoint | ✅ | CQRS: ClaimNextTaskCommand |
| `POST /gradient` submit endpoint | ✅ | Applies → new version → telemetry row |
| CQRS dispatcher (McpServer.Cqrs) | ✅ | One handler per admin page |
| `seed-tasks` CLI (synthetic) | ✅ | Legacy path |
| `seed-real-tasks` CLI | ✅ | Phase-A real-shard seeding |
| `dump-events` CLI | ✅ | Diagnostic log tail |
| Deadline calibration from measured tps | 🟡 | P0 in flight (this commit) — build + verify pending |
| Seed-size feedback loop | ⚪ | P1 — blocked on P0 landing |
| "Stuck but alive" UI counter | ⚪ | P2 |
| Purge of legacy 1,605 seed rows | ⚪ | P4 — product decision pending |

## 5. Coordinator admin UI (Blazor Server)

| Page | Status | Notes |
|---|---|---|
| `/admin/dashboard` | ✅ | 2-s refresh, per-worker grid, live rate |
| `/admin/tasks` | ✅ | Status rollups |
| `/admin/task-browser` | ✅ | Queued/finished split, shard-path badges |
| `/admin/logs` | ✅ | Filter by level + worker |
| `/admin/api-keys` | ✅ | Legacy multi-client rotation |
| `/admin/install` | ✅ | Install landing |
| Viewer-local time (`<time data-utc>`) | ✅ | MutationObserver rewrite |
| Worker-grid click-to-sort + filter | ✅ | Client-side, diff-guarded |
| Infinite-loop freeze in sort indicators | ✅ | Fixed in commit `5347cd9` |

## 6. Worker

| Feature | Status | Notes |
|---|---|---|
| Docker container (linux/amd64) | ✅ | `ghcr.io/sharpninja/bitnet-worker:latest` |
| Boot-time calibration (BenchmarkDotNet) | ✅ | Reports synthetic tok/s |
| `RealTrainingGradient` path | ✅ | Active when `BITNET_ALLOW_SYNTHETIC_SHARDS=false` |
| `CorpusClient` Range streaming | ✅ | int32 token read |
| `FlatParameterPack` serialize | ✅ | |
| Weight download via `/weights/{version}` | ✅ | |
| Gradient submit | ✅ | Observed loss-going-down signal on LEGION2 |
| Heartbeat | ✅ | |
| Measured-backprop telemetry upstream | 🟡 | Coordinator already reads it from gradient_events; no dedicated wire field |

## 7. Fleet nodes

| Node | Role | Status |
|---|---|---|
| PAYTON-DESKTOP | Coordinator (Windows Service) | ✅ online |
| LEGION2 | Worker (Docker) | ✅ running real training |
| DESKTOP | Worker (Docker) | ✅ running real training |

## 8. Source control & CI

| Item | Status | Notes |
|---|---|---|
| Azure DevOps (`azure`) as primary | ✅ | All pushes target `azure` |
| GitHub (`origin`) downstream mirror | ✅ | On-demand sync after `azure` |
| Azure DevOps pipelines doc | ✅ | `docs/azure-devops-pipelines.md` |
| GHCR worker image publish | ✅ | `ghcr-push-worker.ps1` |
| Signed-commit / PR-only bypass | 🟡 | GitHub reports bypass warnings; intentional for mirror |

## 9. Documentation

| Doc | Status |
|---|---|
| README / SUMMARY | ✅ |
| Architecture | ✅ |
| Bucketing guide + impl plan | ✅ |
| DataGen guide | ✅ |
| Implementation plan v3 | ✅ |
| Full implementation plan (real training + benchmarks + purity) | ✅ |
| Real training implementation plan v1.0 | ✅ |
| Benchmarking | ✅ |
| Azure DevOps pipelines | ✅ |
| Releases and packaging | ✅ |
| Repo alignment guidelines | ✅ |
| Usage | ✅ |
| Training and visualization | ✅ |
| Distributed training | ✅ |
| **Training corpus scope (new)** | ✅ |
| **State of completion (this doc)** | ✅ |

## 10. Remaining work — ordered by priority

### P0 — Real-throughput deadline calibration
**Status:** 🟡 code written, build + deploy pending.
**Files:** `SqliteTelemetryStore.GetMeasuredTokensPerSecond`,
`SqliteWorkQueueStore.TryClaimNextPending(Func<long, TimeSpan>)`,
`ClaimNextTaskCommandHandler.LeaseFor`.
**Acceptance:** 40 seeded real-shard tasks complete end-to-end
without claim-expiry loops; `/admin/dashboard` shows monotonically
rising Done count as workers submit.

### P1 — Seed-size feedback loop
**Status:** ⚪ not started. Depends on P0.
**Plan:** `seed-real-tasks` reads mean `tokens/(wall_clock_ms/1000)`
from `gradient_events`, defaults `tokensPerTask ≈ realTps × 600`.
**Acceptance:** newly seeded tasks fit in the 10-minute target
without operator-tuned `tokensPerTask`.

### P2 — "Soft-expired but alive" UI counter
**Status:** ⚪ not started.
**Plan:** Dashboard query joins task rows with worker heartbeat; a
row with `deadline < now` but a fresh heartbeat is "soft-expired".
Display as a second counter next to Assigned.
**Acceptance:** operator can tell a stuck worker from a slow-but-alive
worker at a glance.

### P3 — Script hygiene
**Status:** ⚪ not started.
**Plan:** Delete `.claude/scripts/tmp-*.ps1` that are no longer used;
promote anything still valuable to a non-`tmp-` filename + document.
**Acceptance:** `git status` clean of stray `tmp-*` entries.

### P4 — Legacy `task-seed-*` rows
**Status:** ⚪ product decision pending.
**Plan:** Either hard-delete 1,605 synthetic Done rows so the
dashboard progress reflects real training only, or tag them with a
`legacy=1` column and exclude from the headline counter.
**Acceptance:** dashboard progress bar tracks real-corpus training
signal, not backfilled stubs.

### Further horizon (not sequenced)

- ~~Corpus-v2 scale (200K+ examples) to unlock `truckmate-large`.~~ ✅ Done 2026-04-17.
- `seed-real-tasks --shard-prefix truckmate-v2` flag to target v2 shards in seeding. Currently seeds fall back to legacy behavior.
- Real-ASR-trace ingestion with PII scrubbing.
- Multi-turn corpus generator.
- Per-worker GPU support (currently CPU-only in containers).
- Automated nightly telemetry prune (`D-5` in the full impl plan).

## 11. How to resume after context break

1. `git status` to see whether the P0 diff is still uncommitted.
2. `dotnet build BitNet-b1.58-Sharp.slnx` — confirm P0 compiles.
3. `.claude/scripts/tmp-deploy-coord.ps1` — robocopy + restart service.
4. Open `/admin/dashboard` — watch Assigned ≥ 1 for ~10 min, then
   Done increment.
5. If P0 verified, move to P1: wire measured-tps into
   `seed-real-tasks`.
