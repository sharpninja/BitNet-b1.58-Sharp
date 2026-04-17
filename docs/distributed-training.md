# Distributed training (coordinator + workers)

Snapshot of the fleet architecture, what is landed, what is broken, and
what comes next. Updated 2026-04-17.

## 1. Topology

```
┌────────────────────┐        ┌──────────────────────┐
│  Worker container  │◀──────▶│  Coordinator (ASP)   │
│  LEGION2           │  HTTP  │  PAYTON-DESKTOP:5000 │
│  Docker, linux/amd │        │  Windows Service     │
└────────────────────┘        │  SQLite (WAL)        │
┌────────────────────┐        │  Blazor Server admin │
│  Worker container  │◀──────▶│  /admin/*            │
│  DESKTOP           │  HTTP  │                      │
│  Docker, linux/amd │        └──────────────────────┘
└────────────────────┘
```

- Coordinator hosts the work queue, weight store, worker registry,
  telemetry store, event-log store, and the admin Blazor UI. All state
  lives in one SQLite DB at `F:\ProgramData\BitNetCoordinator\coordinator.db`
  with WAL + `busy_timeout=5000`.
- Workers run as Linux Docker containers pulled from GHCR
  (`ghcr.io/sharpninja/bitnet-worker:latest`). They claim tasks, download
  the weight blob, compute a gradient, and submit it back.
- Auth: one shared `X-Api-Key` header on every worker→coordinator call.
  No OAuth, no per-client secrets. Flipping the env var + restarting
  the coordinator invalidates every worker until redeployed with the
  new key.
- Admin UI auth: OIDC (Duende IdentityServer) with a single admin
  TestUser seeded from `CoordinatorOptions.Admin`.

## 2. What is landed

### Coordinator

- CQRS dispatcher (McpServer.Cqrs) with query handlers behind every
  admin page.
- Persisted stores (one SQLite file, separate tables):
  `SqliteWorkQueueStore`, `SqliteWorkerRegistryStore`,
  `SqliteTelemetryStore`, `SqliteLogStore`, `FileSystemWeightStore`.
- Track 7 hard-reset on weight-version mismatch: if a persisted
  weight's element count disagrees with the expected `flat-param
  length` for the active `ICoordinatorModelConfig`, the coordinator
  logs a banner, reinitializes as `latestVersion + 1`, and retains
  the legacy blob on disk.
- `/corpus/{shardId}` serves the tokenized `int32` stream the worker
  expects. Resolution order is in `Services/CorpusShardLocator.cs`:
  `corpus/tokenized/{id}.bin` → `corpus/{id}.bin` → `corpus/{id}.txt`
  → bare id. Content-Type flips between
  `application/octet-stream` and `text/plain; charset=utf-8`.
- Dev CLI subcommands on the coordinator DLL (all read the service's
  DB path from registry env):
  - `seed-tasks [count] [tokensPerTask]` — synthetic `shard-seed`
    tasks. Used before Phase A landed.
  - `seed-real-tasks [tokensPerTask] [maxPerShard]` — walks
    `corpus/tokenized/*.bin`, slices each shard into chunks of
    `tokensPerTask` int32s, enqueues one task per chunk with the
    real `shardId` + byte offset/length. This is the Phase-A seed
    path.
  - `generate-corpus [count]` — raw text shards.
  - `tokenize-corpus [vocabSize]` — produces the
    `corpus/tokenized/*.bin` int32 streams + `vocab.json`.
  - `dump-events [limit] [minLevel]` — diagnostic dump of the
    worker-log ring.
- Admin pages under `Components/Pages/`:
  - `/admin/dashboard` — fleet progress + live rate + per-worker grid,
    auto-refresh every 2 s over the InteractiveServer circuit.
  - `/admin/tasks` — task list + per-state counts.
  - `/admin/task-browser` — queued/finished split, per-row shard-path
    resolution with `ok`/`warn` styling when a shard is missing.
  - `/admin/logs` — worker-log viewer with filtering.
  - `/admin/api-keys` — admin-only rotation for legacy multi-client
    auth (superseded by single shared key).
  - `/admin/install` — install-script landing page.

### UI primitives (shared via `Components/App.razor`)

- **Local-time rendering.** Every timestamp is emitted as
  `<time data-utc="<ISO>">UTC string</time>`. A MutationObserver in
  `App.razor` rewrites `textContent` to the viewer's local time and
  sets `title` to the UTC original. A `data-localized` guard prevents
  re-processing. Works across the 2-s Blazor patch cycle.
- **Worker grid — filter + sort.** Second MutationObserver-backed
  script exposes `window.bitnetWorkerGrid.applyFilter(state)` and
  click-to-sort on any `th[data-col]`. Sort cycles asc → desc → none,
  visible via `↑`/`↓` in the header. A `state.reordering` guard
  prevents the observer from re-triggering itself; self-mutations are
  diffed against the desired value so unchanged rows don't fire the
  observer.

### Worker

- Runs `RealTrainingGradient` (not the legacy synthetic path) when
  `BITNET_ALLOW_SYNTHETIC_SHARDS=false`.
- `CorpusClient` streams the tokenized shard via Range requests.
- `FlatParameterPack` serializes the full `BitNetTransformer` state
  into a single `float[]` so the gradient wire format stays flat.
- Calibration pass at boot: `BenchmarkDotNet` measures synthetic
  tok/s over ~1 s, multiplies to compute a "target task size" for a
  10-minute task.
- Worker registration sends `CpuThreads`, calibrated tokens/sec, and
  target task size. Coordinator uses these for the Fleet Dashboard
  live rate + ETA.

### Operational scripts (`.claude/scripts/`)

Persistent scripts:

- `ghcr-push-worker.ps1` — builds worker image, tags, pushes to
  GHCR with a fresh `DOCKER_CONFIG` + explicit basic-auth to bypass
  the Windows credsStore.
- `remote-recycle-worker.ps1` — WinRM into a host, read the live
  container's env vars, preserve a curated wanted-list
  (`BITNET_COORDINATOR_URL`, `BITNET_WORKER_ID`, `BITNET_WORKER_NAME`,
  `BITNET_WORKER_API_KEY`, `BITNET_MODEL_PRESET`,
  `BITNET_ALLOW_SYNTHETIC_SHARDS`), `docker stop` + `docker rm`,
  `docker pull`, `docker run` with the same env.
- `desktop-*.ps1` — coordinator lifecycle (build, install service,
  read logs, pull+build, tokenize corpus, seed tasks).

Temporary scripts prefixed `tmp-*` are throwaway; ignore them for
anything durable.

## 3. Current priorities (2026-04-17)

Ordered most urgent first.

### P0: Task deadline must be sized by real backprop throughput, not calibration

**Symptom.** Seeded 40 real-shard tasks. Workers claimed two within a
second, started `RealTrainingGradient`, and the dashboard then showed
0 assigned / 40 pending for minutes. Workers were never idle — the
coordinator had already expired the claim.

**Root cause.** Calibration measures synthetic throughput (`4,750
tok/s` on LEGION2). Real full-transformer backprop runs at `~30
tok/s` on the same box — 160× slower. Sample from the `[FullTrainer]`
line on LEGION2 during a live task:

```
[FullTrainer] Epoch 1/4 | Loss: 6.460942 | Perplexity: 639.66 |
              Sequences: 128 | Tokens: 16,256 | Wall: 548.36s
```

One 16,384-token task × 4 local steps × 548 s/step ≈ 37 minutes, vs
the 10-minute target the worker reported at registration. The claim
expires long before the worker submits.

**Fix direction.**

1. Stop using calibration's synthetic tok/s for deadlining. Let each
   worker submit a *post-hoc* real-throughput telemetry on first
   successful gradient, and use that going forward.
2. Coordinator-side: when assigning a task, compute `deadline =
   now + max(safety, tokens / reportedRealTps + headroom)`. Default
   to a generous 2× multiplier until real tok/s is known.
3. Task sizing at seed time should be driven off the same post-hoc
   throughput, not the synthetic calibration.

### P1: Seed tasks should pick a reasonable initial size

`seed-real-tasks` currently slices every shard at a fixed
`tokensPerTask` passed on the command line. There's no feedback loop
that looks at prior real-throughput telemetry. After P0, the seed
command should default to sizing chunks at `reportedRealTps × 600 s`
so a task fits in the 10-minute target out of the box.

### P2: Dashboard should distinguish "assigned & computing" from "assigned & stuck"

Right now the Fleet Dashboard shows 0 assigned the instant the
claim's deadline passes, even though the worker is still grinding.
Add a secondary count of "soft-expired but alive" rows (heartbeat
fresh, gradient not yet submitted) so operators can tell the
difference.

### P3: Clean up throwaway scripts

Keep `scripts/*.ps1` tidy. The active remote helpers are
`deploy-coord.ps1`, `purge-and-reseed.ps1`, `purge-telemetry.ps1`,
`dump-events.ps1`, and `check-telemetry.ps1`. Anything throwaway
should stay out of source control.

### P4: Investigate the 1605 "Done" tasks

The dashboard shows 1,605 completed tasks from before Phase A landed
(all `task-seed-*`, synthetic 8,192-token chunks). These should be
purged before the first real-data training run so the progress bar
reflects the actual training signal, not backfilled stubs. Decide
whether to hard-delete or keep as historical telemetry.

## 4. Design decisions worth preserving

### One SQLite file for all coordinator state

Five stores, one file, WAL + `busy_timeout=5000`. No separate Postgres
or message broker. Blazor queries go through CQRS handlers that talk
directly to the store classes. This keeps the operational surface
tiny — one file to back up, one file to inspect, one place where
coordinator state lives.

### One shared X-Api-Key

Superseded an earlier per-client OAuth design. One `X-Api-Key` env
var on the coordinator, the same value on every worker. Rotate =
change the env var + restart. Rationale: the fleet is small (≤ 10
workers in practice), operator-owned, and the operational cost of
per-client secrets + refresh flows isn't paid back in this threat
model. Preference captured in `~/.claude/projects/.../memory/`.

### Hard-reset on persisted weight mismatch

When the coordinator boots and finds a persisted weight whose element
count doesn't match `flat-param length` for the active
`ICoordinatorModelConfig`, it:

1. Logs a banner identifying the mismatch
2. Creates a fresh zero-init weight at `latestVersion + 1`
3. Leaves the old blobs on disk untouched

Rationale: silently re-scaling is dangerous; failing hard is noisy
but auditable. The banner shows up at every `/admin/dashboard` load
until a new weight version is pushed.

### Real-time admin UI over the Blazor circuit

InteractiveServer mode, 2-second timer calling `InvokeAsync(async =>
{ await ViewModel.LoadAsync(); StateHasChanged(); })`. No SignalR
hub, no JS polling. Rationale: the Blazor diff reuses existing DOM
nodes so only changed cells repaint, which lets the dashboard stay
responsive under a ~1 Hz telemetry rate without hand-rolled DOM
management.

### `<time data-utc>` + MutationObserver for viewer-local time

Server renders every timestamp as UTC inside a `<time>` tag with
`data-utc="<ISO>"`. One client-side script in `App.razor` rewrites
every such node to the viewer's local time. Re-runs on every Blazor
patch. Guarded by `data-localized="1"` so no node is processed twice.
Rationale: we can't know the viewer's tz at render time, and round-
tripping through a JS interop call per cell would be expensive.

### CorpusShardLocator prefers tokenized/*.bin

Introduced when the `/corpus/{id}` route needed to stream the int32
tokens the worker's `CorpusClient` reads. Preferred order:
`corpus/tokenized/{id}.bin` → `corpus/{id}.bin` → `corpus/{id}.txt`
→ bare id. Legacy text shards still resolve if a run predates
tokenization.

### CQRS ViewModel per admin page

Every page injects a `partial class` ViewModel built with
`CommunityToolkit.Mvvm` `[ObservableProperty]` and dispatches queries
through `IDispatcher`. Keeps data-access and rendering separate and
makes it trivial to add a new admin page without touching existing
pipelines.

### Worker grid sort/filter is purely client-side

Every row carries `data-<col>` attributes. A MutationObserver
re-applies the current filter + sort after every Blazor patch.
Rationale: sending sort state over the circuit would turn every
sort click into a server round-trip and a full snapshot refetch.
DOM reordering under a 2-s refresh is cheap; the server stays
stateless w.r.t. per-session view preferences.

## 5. Pointers

- Coordinator host: `src/BitNetSharp.Distributed.Coordinator/Program.cs`
- Worker entrypoint: `src/BitNetSharp.Distributed.Worker/Program.cs`
- Shared contracts: `src/BitNetSharp.Distributed.Contracts/`
- Real-training gradient (worker): `src/BitNetSharp.Distributed.Worker/RealTrainingGradient.cs`
- Weight apply: `src/BitNetSharp.Distributed.Coordinator/Services/WeightApplicationService.cs`
- Admin layout: `src/BitNetSharp.Distributed.Coordinator/Components/MainLayout.razor`
- Auth: `src/BitNetSharp.Distributed.Coordinator/Authentication/` +
  `Program.cs` (`WorkerApiKeyPolicy`, `AdminPolicy`)
