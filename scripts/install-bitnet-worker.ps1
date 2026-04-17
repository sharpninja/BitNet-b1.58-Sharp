# ─────────────────────────────────────────────────────────────────────
#  install-bitnet-worker.ps1 — generic one-shot worker launcher.
#
#  Pulls ghcr.io/sharpninja/bitnetsharp-worker:latest and runs it with
#  the hardened `docker run` flags used in production. Reads all
#  configuration from environment variables or command-line arguments.
#
#  Required environment variables (or matching -Arguments):
#    BITNET_COORDINATOR_URL   Coordinator base URL (e.g. https://host:5001/)
#    BITNET_WORKER_API_KEY    Shared API key set by the operator on the
#                             coordinator via Coordinator__WorkerApiKey
#
#  Optional:
#    BITNET_WORKER_ID         Stable worker identity (default: $env:COMPUTERNAME)
#    BITNET_WORKER_NAME       Human-friendly display name (default: $env:COMPUTERNAME)
#    BITNET_CPU_THREADS       Hard cap on threads (default: all logical CPUs)
#    BITNET_HEARTBEAT_SECONDS Heartbeat cadence, seconds (default: 30)
#    BITNET_LOG_LEVEL         Serilog minimum level (default: info)
#    BITNET_IMAGE             Container image:tag  (default: ghcr.io/sharpninja/bitnetsharp-worker:latest)
#    BITNET_CONTAINER_NAME    Container name        (default: bitnet-worker)
#
#  Example:
#    $env:BITNET_COORDINATOR_URL = 'https://bitnet.example.com:5001/'
#    $env:BITNET_WORKER_API_KEY  = 'the-secret-the-operator-set'
#    pwsh -NoProfile -ExecutionPolicy Bypass -File .\install-bitnet-worker.ps1
# ─────────────────────────────────────────────────────────────────────

[CmdletBinding()]
param(
    [string]$CoordinatorUrl = $env:BITNET_COORDINATOR_URL,
    [string]$ApiKey         = $env:BITNET_WORKER_API_KEY,
    [string]$WorkerId       = $env:BITNET_WORKER_ID,
    [string]$WorkerName     = $env:BITNET_WORKER_NAME,
    [string]$CpuThreads     = $env:BITNET_CPU_THREADS,
    [string]$HeartbeatSeconds = $env:BITNET_HEARTBEAT_SECONDS,
    [string]$LogLevel       = $env:BITNET_LOG_LEVEL,
    [string]$Image          = $(if ($env:BITNET_IMAGE) { $env:BITNET_IMAGE } else { 'ghcr.io/sharpninja/bitnetsharp-worker:latest' }),
    [string]$ContainerName  = $(if ($env:BITNET_CONTAINER_NAME) { $env:BITNET_CONTAINER_NAME } else { 'bitnet-worker' })
)

$ErrorActionPreference = 'Stop'

function Fail([string]$msg) {
    Write-Host "[installer] ERROR: $msg" -ForegroundColor Red
    exit 2
}

if ([string]::IsNullOrWhiteSpace($CoordinatorUrl)) {
    Fail 'BITNET_COORDINATOR_URL is not set. Export it or pass -CoordinatorUrl.'
}
if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    Fail 'BITNET_WORKER_API_KEY is not set. Export it or pass -ApiKey. Ask the operator for the value set on Coordinator__WorkerApiKey.'
}

if ([string]::IsNullOrWhiteSpace($WorkerId))   { $WorkerId   = $env:COMPUTERNAME }
if ([string]::IsNullOrWhiteSpace($WorkerName)) { $WorkerName = $env:COMPUTERNAME }
if ([string]::IsNullOrWhiteSpace($HeartbeatSeconds)) { $HeartbeatSeconds = '30' }
if ([string]::IsNullOrWhiteSpace($LogLevel)) { $LogLevel = 'info' }

Write-Host "[installer] Using Docker worker image." -ForegroundColor Cyan
Write-Host "[installer] coordinator: $CoordinatorUrl"
Write-Host "[installer] worker id  : $WorkerId"
Write-Host "[installer] image      : $Image"

# Stop + remove any existing container with the same name so the
# re-run picks up the freshly-pulled image cleanly.
docker rm -f $ContainerName 2>$null | Out-Null

$envArgs = @(
    '-e', "BITNET_COORDINATOR_URL=$CoordinatorUrl",
    '-e', "BITNET_WORKER_API_KEY=$ApiKey",
    '-e', "BITNET_WORKER_ID=$WorkerId",
    '-e', "BITNET_WORKER_NAME=$WorkerName",
    '-e', "BITNET_HEARTBEAT_SECONDS=$HeartbeatSeconds",
    '-e', "BITNET_LOG_LEVEL=$LogLevel"
)
if (-not [string]::IsNullOrWhiteSpace($CpuThreads)) {
    $envArgs += @('-e', "BITNET_CPU_THREADS=$CpuThreads")
}

$dockerArgs = @(
    'run',
    '-d',
    '--pull=always',
    '--name', $ContainerName,
    '--restart', 'unless-stopped',
    '--read-only',
    '--tmpfs', '/tmp:size=64m,mode=1777',
    '--cap-drop', 'ALL',
    '--security-opt', 'no-new-privileges:true'
) + $envArgs + @($Image)

Write-Host "[installer] docker $($dockerArgs -join ' ')" -ForegroundColor DarkGray
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    Fail "docker run failed with exit code $LASTEXITCODE"
}

Write-Host "[installer] Worker container '$ContainerName' started. Tail logs with:" -ForegroundColor Green
Write-Host "             docker logs -f $ContainerName"
