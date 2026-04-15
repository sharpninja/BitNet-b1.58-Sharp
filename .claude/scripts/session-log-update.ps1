# Helper that (re)creates / updates the MCP session log for the current
# BitNet distributed training work. Safe to re-run — New-McpSessionLog is
# idempotent on the server when the same -SessionId is supplied.

param(
    [switch]$SkipCreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Import-Module 'F:\GitHub\McpServer\tools\powershell\McpSession.psm1' -Force

$null = Initialize-McpSession `
    -Agent Claude `
    -Model claude-opus-4-6 `
    -MarkerPath 'F:\GitHub\McpServer\AGENTS-README-FIRST.yaml'

$sessionId = 'Claude-20260415T120000Z-bitnet-distributed-training'

if (-not $SkipCreate) {
    $session = New-McpSessionLog `
        -SourceType Claude `
        -SessionId $sessionId `
        -Title 'BitNet distributed CPU training system + probability floor fix' `
        -Model claude-opus-4-6
    Write-Host "created sessionId=$($session.sessionId)"
}

# ── Turn 1: probability floor fix ───────────────────────────────────────
Add-McpSessionTurn `
    -QueryTitle 'Unify BitNet probability floor to 1e-6 across perplexity paths' `
    -QueryText  'Apply and commit the outstanding probability floor fixes so BitNet perplexity comparisons against the traditional baseline are apples-to-apples.' `
    -Interpretation 'BitNetPaperModel.CalculatePerplexity and BitNetPaperAudit both used a 1e-9 probability floor while TraditionalLocalModel uses 1e-6, adding up to 6.9 nats per floor-hit token to BitNet perplexity. Align all three code paths to 1e-6.' `
    -Response 'Committed as ae8ee29. Updated ProbabilityFloor const in BitNetPaperModel.cs and BitNetPaperAudit.cs plus two inline literals in the audit softmax path. 2 files changed, 9 insertions, 4 deletions. Observed impact on prior benchmark: WikiText2 audit 19661 -> 16444, C4 66957 -> 16533, RedPajama 19576 -> 9333.' `
    -Status completed `
    -Tags @('bitnet','perplexity','benchmark','bug-fix') `
    -FilesModified @('src/BitNetSharp.Core/BitNetPaperModel.cs','src/BitNetSharp.Core/BitNetPaperAudit.cs') `
    -DesignDecisions @('Probability floor standardized to 1e-6 for all perplexity paths.','No change to training logic — floor only affects evaluation.')

# ── Turn 2: lock v1 defaults for distributed training system ────────────
Add-McpSessionTurn `
    -QueryTitle 'Lock v1 defaults for distributed training system' `
    -QueryText  'Coordinator hosted on PAYTON-DESKTOP behind ngrok. Pick a database engine for the work queue and weight versions.' `
    -Interpretation 'Single coordinator box on PAYTON-DESKTOP with ngrok exposure. Work queue is a relational problem (atomic dequeue, state transitions). SQLite WAL fits the single-writer topology with zero ops.' `
    -Response 'Locked: coordinator=PAYTON-DESKTOP, exposure=ngrok reserved domain (Basic plan for rate limit + bandwidth), auth=bearer token + pre-shared enrollment key, DB=SQLite WAL + filesystem blob store, gradient format=int8 with error feedback, local SGD K=4, work unit=512 tokens per task (baseline, overridden per worker by capability report), max staleness=10, corpus strategy=coordinator-streamed shards, checkpoint cadence=60s.' `
    -Status completed `
    -Tags @('distributed','architecture','decision','v1') `
    -DesignDecisions @('SQLite WAL chosen over Postgres/Mongo/Redis for single-coordinator topology.','ngrok Basic plan required for rate limit + bandwidth.','Bearer token + pre-shared enrollment key for worker auth.','int8 with error feedback as gradient compression default.','Coordinator streams corpus shards, workers hold no persistent state.')

# ── Turn 3: worker + docker scaffolding with BDN self-calibration ───────
Add-McpSessionTurn `
    -QueryTitle 'Add distributed worker with Docker image and BenchmarkDotNet self-calibration' `
    -QueryText  'Scaffold BitNetSharp.Distributed.Worker plus a Docker container so workers can be spun up as instances. Each worker runs a BenchmarkDotNet capability pass on startup and the coordinator sizes work units to ~10 minutes of compute on that specific box.' `
    -Interpretation 'Phase D-1 worker scaffold. Fail-fast env config, BDN InProcessNoEmitToolchain so calibration works in minimal runtime container, CapabilityReport with RecommendedTokensPerTask math (tok/s x 600s x efficiency rounded to 512-token granularity), health beacon for Docker healthcheck, graceful SIGTERM.' `
    -Response 'Committed as 3d2600a (16 files, 1042 insertions). Worker project scaffolded with WorkerConfig, WorkerCapabilityBenchmark, StartupCalibrator, CapabilityReport, HealthBeacon, Program.cs, AssemblyInfo (InternalsVisibleTo tests). Docker image: multi-stage sdk:10.0 -> runtime:10.0, non-root uid 10001, HEALTHCHECK via beacon file mtime, 120s start-period. docker-compose.yml scales via --scale worker=N, read-only FS, tmpfs /tmp, cap_drop ALL. build.ps1 tags with git short SHA. 8 xunit tests for CapabilityReport math (linear scaling, efficiency override, clamping, 512-token rounding, fallback, display string). Smoke test on PAYTON-LEGION2: 8.3s calibration, 3,155 tok/s on 16 threads, recommended 473,600 tokens per task for 10-min target. Solution and test project references updated.' `
    -Status completed `
    -Tags @('distributed','worker','docker','benchmark','phase-d1') `
    -FilesModified @(
        'src/BitNetSharp.Distributed.Worker/BitNetSharp.Distributed.Worker.csproj',
        'src/BitNetSharp.Distributed.Worker/WorkerConfig.cs',
        'src/BitNetSharp.Distributed.Worker/CapabilityReport.cs',
        'src/BitNetSharp.Distributed.Worker/WorkerCapabilityBenchmark.cs',
        'src/BitNetSharp.Distributed.Worker/StartupCalibrator.cs',
        'src/BitNetSharp.Distributed.Worker/HealthBeacon.cs',
        'src/BitNetSharp.Distributed.Worker/Program.cs',
        'src/BitNetSharp.Distributed.Worker/AssemblyInfo.cs',
        'docker/worker/Dockerfile',
        'docker/worker/Dockerfile.dockerignore',
        'docker/worker/docker-compose.yml',
        'docker/worker/.env.example',
        'docker/worker/build.ps1',
        'BitNet-b1.58-Sharp.slnx',
        'tests/BitNetSharp.Tests/BitNetSharp.Tests.csproj',
        'tests/BitNetSharp.Tests/CapabilityReportTests.cs') `
    -DesignDecisions @(
        'Worker self-calibrates on startup using BenchmarkDotNet with InProcessNoEmitToolchain (no child-process spawning in container).',
        'CapabilityReport.RecommendedTokensPerTask scales tok/s x 600s x efficiency (0.25 default) rounded up to 512-token granularity.',
        'Backward+optimizer+serialize overhead folded into efficiency multiplier, not into the calibration workload itself.',
        'Health beacon is a touched file under /tmp — survives tmpfs read-only root, gives orchestrator true progress signal.',
        'Container non-root uid 10001, cap_drop ALL, read_only FS, tmpfs /tmp, no-new-privileges.') `
    -RequirementsDiscovered @(
        'Phase D-1 will replace the stub [register] log line with real HTTP POST to coordinator /register.',
        'Phase D-1 will add /work long-poll and gradient submission after calibration.',
        'ngrok Basic plan or higher required because Free tier 40 req/min starves any meaningful worker pool.')

# ── Turn 4: session log bootstrap ───────────────────────────────────────
Add-McpSessionTurn `
    -QueryTitle 'Bootstrap MCP session log for this BitNet work' `
    -QueryText  'Dont forget to keep session log up to date.' `
    -Interpretation 'User reminded me to persist this conversation into the MCP session log. Created session Claude-20260415T120000Z-bitnet-distributed-training and backfilled three turns covering the probability floor fix commit, the distributed system v1 default lock, and the worker + Docker container scaffold with BDN self-calibration.' `
    -Response 'Session created, three completed turns appended, helper script saved to .claude/scripts/session-log-update.ps1 for future re-runs.' `
    -Status completed `
    -Tags @('session-log','housekeeping')

Write-Host 'session log updated'
