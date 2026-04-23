#requires -Version 7
<#
.SYNOPSIS
    Stop any running Ollama daemon and start the BitNetSharp Ollama-compat server.

.DESCRIPTION
    1. Stops the `ollama` service if installed, kills tray + daemon processes,
       and releases whatever owns the target port.
    2. Starts `dotnet run --project src/BitNetSharp.App -- serve` with the
       provided model file in the foreground.

    Run from the repo root (or worktree root). Ctrl+C stops the serve process
    cleanly; at that point the original Ollama daemon is NOT auto-restarted.

.PARAMETER Model
    Path to a BitNetSharp-native .gguf file. Relative or absolute.

.PARAMETER Port
    TCP port to listen on. Defaults to 11434 (same as real Ollama).

.PARAMETER BindHost
    Interface to bind to. Defaults to 127.0.0.1. Use 0.0.0.0 for LAN access.

.PARAMETER Configuration
    Build configuration passed to `dotnet run`. Defaults to Release.

.PARAMETER NoCors
    Pass --no-cors to the server.

.EXAMPLE
    pwsh scripts/start-ollama-compat.ps1 -Model data/models/bonsai.bitnetsharp.gguf

.EXAMPLE
    pwsh scripts/start-ollama-compat.ps1 -Model data/models/x.gguf -Port 11435 -BindHost 0.0.0.0
#>
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 11434,
    [string]$BindHost = '127.0.0.1',
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Release',
    [switch]$NoCors
)

$ErrorActionPreference = 'Stop'

function Write-Section([string]$Title) {
    Write-Host ''
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

# 1. Stop Ollama Windows service (if installed).
Write-Section 'Stopping Ollama service (if present)'
$svc = Get-Service -Name 'Ollama' -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne 'Stopped') {
        Write-Host "  Stopping service: $($svc.Name) ($($svc.Status))"
        Stop-Service -Name 'Ollama' -Force -ErrorAction Stop
    } else {
        Write-Host '  Service already stopped.'
    }
} else {
    Write-Host '  No Ollama service registered.'
}

# 2. Kill ollama processes (app, tray, daemon).
Write-Section 'Killing ollama processes'
$procs = Get-Process -Name 'ollama*' -ErrorAction SilentlyContinue
if ($procs) {
    foreach ($p in $procs) {
        Write-Host "  Killing $($p.ProcessName) (PID $($p.Id))"
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
} else {
    Write-Host '  No ollama processes running.'
}

# 3. Free the target port: whatever holds it, kill it.
Write-Section "Releasing port $Port"
$retries = 10
while ($retries -gt 0) {
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $conn) {
        Write-Host "  Port $Port is free."
        break
    }
    $pidHolding = $conn.OwningProcess
    $holder = Get-Process -Id $pidHolding -ErrorAction SilentlyContinue
    $name = if ($holder) { $holder.ProcessName } else { '<exited>' }
    Write-Host "  Port $Port held by PID $pidHolding ($name); killing."
    Stop-Process -Id $pidHolding -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    $retries--
}
if ($retries -eq 0) {
    throw "Port $Port could not be released after 10 attempts."
}

# 4. Sanity: BitNetSharp.App project must exist.
Write-Section 'Checking repo'
$projPath = 'src/BitNetSharp.App/BitNetSharp.App.csproj'
if (-not (Test-Path $projPath)) {
    throw "Not in a BitNetSharp repo root (looked for $projPath). cd to the repo or worktree root first."
}
if (-not (Test-Path $Model)) {
    Write-Host "  Warning: model path '$Model' does not exist yet; passing through to serve anyway."
}

# 5. Start the BitNetSharp Ollama-compat server in the foreground.
Write-Section "Starting BitNetSharp serve on ${BindHost}:${Port}"
$dotnetArgs = @(
    'run', '--project', $projPath, '-c', $Configuration, '--',
    'serve',
    "--host=$BindHost",
    "--port=$Port",
    "--model=$Model"
)
if ($NoCors) { $dotnetArgs += '--no-cors' }

Write-Host "  dotnet $($dotnetArgs -join ' ')"
Write-Host '  (Ctrl+C to stop)'
Write-Host ''
& dotnet @dotnetArgs
$exit = $LASTEXITCODE
Write-Host ''
Write-Host "BitNetSharp serve exited with code $exit"
exit $exit
