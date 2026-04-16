# BitNet-b1.58-Sharp Distributed Training — Session Handoff

**Date:** 2026-04-16
**Session:** `Claude-20260415T120000Z-bitnet-distributed-training`
**HEAD:** `497bef5` (both origin + azure synced)
**Tests:** 307+ fast-lane passing (298 last full run + 9 tokenizer)

## What was built

A complete distributed CPU training system for the BitNet b1.58 ternary
SLM, targeting the **Truck Mate** voice-assistant intent-classification
use case. The system spans four .NET projects, a Docker image, and a
Windows service deployment on two machines.

### Project layout

```
src/
  BitNetSharp.Distributed.Contracts/    # Wire-format DTOs, codecs, tokenizer
  BitNetSharp.Distributed.Coordinator/  # ASP.NET Core host (Duende IS + Blazor)
  BitNetSharp.Distributed.Worker/       # Console app with BDN calibration + Serilog
docker/
  worker/                               # Dockerfile + docker-compose + build.ps1
.claude/
  scripts/                              # PS remoting deployment scripts for PAYTON-DESKTOP
tests/
  BitNetSharp.Tests/                    # 307+ xunit cases
```

### Coordinator (`BitNetSharp.Distributed.Coordinator`)

- **Hosting:** ASP.NET Core Web + `UseWindowsService` — runs as console or Windows service
- **Auth:** Duende IdentityServer 7.4.7 — worker machine-login (client_credentials), admin OIDC (code+PKCE)
- **Persistence:** SQLite WAL — 5 stores (WorkQueue, WorkerRegistry, ClientRevocation, Telemetry, LogStore)
- **Weights:** `FileSystemWeightStore` — immutable versioned fp32 blobs with SHA-256 sidecars
- **Weight apply:** `WeightApplicationService` — in-memory global fp32 vector, staleness compensation (`lr / (1 + staleness * α)`), max-staleness rejection, persist-on-every-apply
- **CQRS:** McpServer.Cqrs library (cross-repo ProjectReference to `F:\GitHub\McpServer`) — `IDispatcher`, `Result<T>` monad, assembly-scanned handlers
- **MVVM:** CommunityToolkit.Mvvm ObservableObject ViewModels, minimal Razor code-behind
- **Background services:** `StaleSweeperService` (stale workers → Gone, timed-out tasks → Pending), `TelemetryPruneService` (hourly DELETE of old rows)
- **Codecs:** `Int8GradientCodec` (per-tensor scale + error-feedback residual), `WeightBlobCodec` (version + fp32 vector)
- **Corpus:** `TruckMateCorpusGenerator` (50K synthetic intent examples), `WordLevelTokenizer` (5174-vocab word-level, pre-tokenized to binary int32 shards)

**Blazor admin pages (all OIDC cookie-gated):**
| Page | URL | Features |
|------|-----|----------|
| Dashboard | `/admin/dashboard` | Interactive server-render, 5s auto-refresh, per-worker table with Drain/Gone/Rotate actions, task counts, weight version, telemetry rollup |
| API keys | `/admin/api-keys` | List/rotate worker OAuth secrets with immediate JWT revocation |
| Tasks | `/admin/tasks` | Queue snapshot + bulk seed form |
| Install | `/admin/install` | Per-client bash + PowerShell worker bootstrap scripts |
| Logs | `/admin/logs` | Structured log viewer with worker/level/search filtering |
| Login | `/Account/Login` | Duende IS interactive login |

**REST endpoints:**
| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/connect/token` | Public | OAuth client_credentials → JWT |
| POST | `/register` | JWT | Worker registration + capability report |
| GET | `/work` | JWT | Atomic task claim (204 when empty) |
| POST | `/heartbeat` | JWT | Worker keep-alive |
| POST | `/gradient` | JWT | Task completion + gradient decode/apply |
| POST | `/logs` | JWT | Structured log ingestion |
| GET | `/weights/{version}` | JWT | Weight blob download with range support |
| GET | `/corpus/{shardId}` | JWT | Corpus shard download |
| GET | `/health` | Public | Health check |
| GET | `/status` | Public | Queue + worker counts JSON |

**CLI subcommands:**
- `seed-tasks [count]` — inject pending tasks into SQLite queue
- `generate-corpus [count]` — produce synthetic Truck Mate training examples
- `tokenize-corpus [maxVocab]` — train tokenizer + write binary int32 shards

### Worker (`BitNetSharp.Distributed.Worker`)

- **Calibration:** BenchmarkDotNet InProcessNoEmitToolchain on startup — measures int8×ternary matmul throughput, reports tokens/sec
- **Task sizing:** `CapabilityReport.RecommendedTokensPerTask()` scales to 10-minute target per worker
- **HTTP client:** `CoordinatorClient` with JWT token cache, auto-refresh, fire-and-forget retry
- **Logging:** Serilog dual-sink (Console + `CoordinatorLogSink` batching to POST /logs)
- **Gradient:** D-4b int8 error-feedback encoding with cross-step residual accumulation
- **Docker:** Multi-stage `mcr.microsoft.com/dotnet/runtime:10.0`, non-root uid 10001, HEALTHCHECK via beacon file mtime, `docker-compose.yml` with `--scale worker=N`

### Deployment (Phase D-2 proven)

- **Coordinator:** Windows service `BitNetCoordinator` on PAYTON-DESKTOP (Ryzen 7 2700X, 16 threads, 32GB)
  - `http://192.168.1.77:5000` (LAN IPv4)
  - DB: `F:\ProgramData\BitNetCoordinator\coordinator.db`
  - Corpus: `F:\ProgramData\BitNetCoordinator\corpus/` (50K text + 10 tokenized binary shards)
  - Env vars in `HKLM\SYSTEM\CurrentControlSet\Services\BitNetCoordinator\Environment`
- **Worker:** Console process on PAYTON-LEGION2 (Ryzen 9 5900HX, 16 threads, 24GB)
  - Calibrates at ~4,750 tok/s
  - Full lifecycle proven: JWT → register → heartbeat → work → gradient → task Done

### Probability floor fix

Commit `ae8ee29` aligned all three perplexity code paths (BitNetPaperModel, BitNetPaperAudit ×2) to `1e-6` matching TraditionalLocalModel. Impact: WikiText2 audit 19661→16444, C4 66957→16533, RedPajama 19576→9333.

## What's next (Phase A — real training)

### Blockers before Truck Mate training can start

1. **Scale BitNetSharp.Core model config**
   - Current: VocabSize=68, ~4.5M params
   - Target: VocabSize=5174, hidden=512, layers=12-16, ~100-150M params
   - Files: `BitNetPaperModel.cs` config struct, `BitNetPaperModelConfig` or similar
   - Risk: scaling may surface numerical issues in the ternary quantization path

2. **Worker corpus loader**
   - Download tokenized `.bin` shards from coordinator via `GET /corpus/{shardId}`
   - Parse int32 sequences into batches of (input, target) pairs
   - Feed into BitNet forward pass
   - Files: new `CorpusDataLoader` in Worker or Core

3. **Replace D-4b synthetic gradient with real backprop**
   - Worker's `RunWorkLoopAsync` currently generates fake gradients
   - Swap for: load weights → forward on corpus batch → backward → encode gradient
   - Files: `Worker/Program.cs` work loop + integration with `BitNetPaperModel.Train`

4. **Convergence sanity check**
   - Seed 50K-example corpus as tasks
   - Run 1-3 epochs of distributed training
   - Verify loss descends in the dashboard telemetry

### Nice-to-haves deferred

- ngrok tunnel setup for external workers
- Admin client_credentials grant for scripted task seeding (current: OIDC-only + CLI)
- Blazor interactive-server upgrade for log viewer (dashboard already upgraded)
- Antiforgery tokens on login + admin POST forms
- CSRF hardening audit

## Credentials (regenerated each install)

Credentials rotate on every `desktop-install-service-only.ps1` run. The latest set is printed by the install script's output. The admin page at `/admin/api-keys` shows the current worker client secrets after OIDC login.

## Key architectural decisions

1. SQLite WAL for all coordinator persistence — single-writer topology, zero ops
2. Duende IdentityServer for both worker machine-login and admin OIDC — one auth provider
3. McpServer.Cqrs cross-repo ProjectReference — MVVM+CQRS enforced, all handlers assembly-scanned
4. Int8 + per-tensor scale gradient codec with error-feedback residual — not ternary, because staleness effect dominates quantization term
5. Staleness compensation: `effective_lr = base_lr / (1 + staleness * α)` with hard reject beyond MaxStalenessSteps
6. Worker self-calibration via BenchmarkDotNet — coordinator sizes tasks to 10-minute target per worker
7. Word-level tokenizer (not BPE) — 5174 vocab is sufficient for the narrow trucking intent domain
8. Static SSR Blazor for most pages, Interactive Server for dashboard only — minimizes SignalR overhead

## How to resume

```powershell
# On PAYTON-LEGION2 (dev box):
cd F:\GitHub\BitNet-b1.58-Sharp
dotnet build BitNet-b1.58-Sharp.slnx -c Release
dotnet test tests/BitNetSharp.Tests -c Release -f net10.0 --filter "Category!=SlowLane"

# Coordinator service on PAYTON-DESKTOP:
pwsh .claude/scripts/desktop-install-service-only.ps1

# Generate + tokenize corpus:
pwsh .claude/scripts/desktop-stage-corpus.ps1
pwsh .claude/scripts/desktop-tokenize-corpus.ps1

# Seed tasks + run worker:
pwsh .claude/scripts/desktop-seed-tasks-cli.ps1 -Count 100
$env:BITNET_COORDINATOR_URL = "http://192.168.1.77:5000/"
$env:BITNET_CLIENT_ID = "<from install output>"
$env:BITNET_CLIENT_SECRET = "<from install output>"
dotnet run --project src/BitNetSharp.Distributed.Worker -c Release -f net10.0
```

## MCP session log

Session `Claude-20260415T120000Z-bitnet-distributed-training` on MCP server at `http://PAYTON-LEGION2:7147`. ~20 turns logged covering every commit and design decision.
