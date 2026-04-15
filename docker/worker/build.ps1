<#
.SYNOPSIS
    Builds (and optionally pushes) the BitNetSharp.Distributed.Worker
    container image. Designed to be callable from the repository root as
    `./docker/worker/build.ps1 -Tag dev`.

.PARAMETER Tag
    Image tag to apply. Defaults to the short git SHA of HEAD so every
    build is traceable back to a commit.

.PARAMETER Registry
    Fully-qualified registry prefix. Defaults to ghcr.io/sharpninja so the
    final image ends up as ghcr.io/sharpninja/bitnetsharp-worker:<Tag>.

.PARAMETER Push
    Push the image to $Registry after building.

.PARAMETER NoCache
    Disable docker build cache (use when dependencies or SDKs change).

.EXAMPLE
    ./docker/worker/build.ps1
    # Tags ghcr.io/sharpninja/bitnetsharp-worker:<short-sha>

.EXAMPLE
    ./docker/worker/build.ps1 -Tag dev

.EXAMPLE
    ./docker/worker/build.ps1 -Tag 0.6.0 -Push
#>
param(
    [string]$Tag,
    [string]$Registry = 'ghcr.io/sharpninja',
    [switch]$Push,
    [switch]$NoCache
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ── Resolve repo root ────────────────────────────────────────────────────
$ScriptDir = $PSScriptRoot
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path

# ── Resolve tag ──────────────────────────────────────────────────────────
if (-not $Tag) {
    Push-Location $RepoRoot
    try {
        $Tag = (git rev-parse --short HEAD).Trim()
    } finally {
        Pop-Location
    }
    if (-not $Tag) { $Tag = 'dev' }
}

$ImageName = "$Registry/bitnetsharp-worker:$Tag"
$DockerfilePath = Join-Path $ScriptDir 'Dockerfile'

Write-Host ""
Write-Host "── BitNetSharp Worker image build ─────────────────────────────" -ForegroundColor Cyan
Write-Host " Repo root : $RepoRoot"
Write-Host " Dockerfile: $DockerfilePath"
Write-Host " Image     : $ImageName"
Write-Host " Push      : $Push"
Write-Host " NoCache   : $NoCache"
Write-Host "───────────────────────────────────────────────────────────────" -ForegroundColor Cyan
Write-Host ""

# ── Build ────────────────────────────────────────────────────────────────
$buildArgs = @(
    'build',
    '-f', $DockerfilePath,
    '-t', $ImageName
)
if ($NoCache) { $buildArgs += '--no-cache' }
$buildArgs += $RepoRoot

Write-Host "docker $($buildArgs -join ' ')" -ForegroundColor DarkGray
& docker @buildArgs
if ($LASTEXITCODE -ne 0) {
    throw "docker build failed (exit $LASTEXITCODE)"
}

Write-Host ""
Write-Host "Built $ImageName" -ForegroundColor Green

# ── Push ─────────────────────────────────────────────────────────────────
if ($Push) {
    Write-Host ""
    Write-Host "Pushing $ImageName …" -ForegroundColor Cyan
    & docker push $ImageName
    if ($LASTEXITCODE -ne 0) {
        throw "docker push failed (exit $LASTEXITCODE)"
    }
    Write-Host "Pushed $ImageName" -ForegroundColor Green
}
