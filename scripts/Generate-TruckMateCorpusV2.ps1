<#
.SYNOPSIS
  Rollout wrapper for truckmate-v2 (200K-example) synthetic corpus.

.DESCRIPTION
  Invokes the coordinator CLI on a remote host to regenerate +
  retokenize the TruckMate corpus at v2 scale. Preserves the
  existing v1 vocab.json as vocab.v1.json before writing the v2
  vocab, so a rollback restores v1-compatible weights. Honors the
  5174 vocab cap — aborts + restores v1 vocab if the cap is
  exceeded.

.PARAMETER Coordinator
  Remote Windows host running BitNetCoordinator service.

.PARAMETER RepoRoot
  Root of the BitNet-b1.58-Sharp clone on the remote host. The
  coordinator DLL is resolved at:
  {RepoRoot}\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0

.PARAMETER DataRoot
  Path on the remote host that contains coordinator.db. The corpus
  directory resolves to {DataRoot}\corpus. Defaults to the value
  the service registers via BITNET_COORDINATOR_DatabasePath in
  its Environment registry key if not supplied.

.PARAMETER TargetCount
  Number of synthetic examples to generate. Default 200000.

.PARAMETER ShardCount
  Shard split. Default 20 (10000 examples per shard).

.PARAMETER Seed
  Deterministic RNG seed. Default 42.

.PARAMETER Name
  Manifest name + shard-ID prefix. Default "truckmate-v2".

.PARAMETER SkipDeploy
  Skip the robocopy + service restart step. Use when the remote
  host already has the target DLL in place.

.PARAMETER SkipSeed
  Skip enqueuing v2 work tasks after tokenization.

.EXAMPLE
  .\Generate-TruckMateCorpusV2.ps1 -Coordinator PAYTON-DESKTOP `
      -RepoRoot 'F:\GitHub\BitNet-b1.58-Sharp' `
      -DataRoot 'F:\ProgramData\BitNetCoordinator'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Coordinator,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [string]$DataRoot,

    [int]$TargetCount = 200000,
    [int]$ShardCount  = 20,
    [int]$Seed        = 42,
    [string]$Name     = 'truckmate-v2',

    [switch]$SkipDeploy,
    [switch]$SkipSeed
)

$ErrorActionPreference = 'Stop'

if ($TargetCount -le 0 -or $ShardCount -le 0) {
    throw "TargetCount and ShardCount must be positive"
}
$examplesPerShard = [int]([math]::Ceiling($TargetCount / [double]$ShardCount))

$remoteDll = Join-Path $RepoRoot 'src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'

function Resolve-RemoteDataRoot {
    param([string]$Host)
    Invoke-Command -ComputerName $Host -ScriptBlock {
        $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
        $env = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
        $dbLine = $env | Where-Object { $_ -match '^BITNET_COORDINATOR_DatabasePath=' }
        if (-not $dbLine) { return $null }
        $dbPath = $dbLine -replace '^BITNET_COORDINATOR_DatabasePath=', ''
        return (Split-Path -Parent $dbPath)
    }
}

if (-not $DataRoot) {
    Write-Host "==> Resolving remote data root from service registry..."
    $DataRoot = Resolve-RemoteDataRoot -Host $Coordinator
    if (-not $DataRoot) {
        throw "Could not resolve DataRoot from service registry on $Coordinator. Pass -DataRoot explicitly."
    }
    Write-Host "    DataRoot = $DataRoot"
}

function Invoke-Coord {
    param([string[]]$CoordArgs)
    Invoke-Command -ComputerName $Coordinator -ScriptBlock {
        param($dll, $CoordArgs)
        $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
        $envStrings = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
        foreach ($line in $envStrings) {
            $eq = $line.IndexOf('=')
            if ($eq -gt 0) {
                Set-Item -Path "env:$($line.Substring(0,$eq))" -Value $line.Substring($eq+1)
            }
        }
        & dotnet $dll @CoordArgs 2>&1
    } -ArgumentList $remoteDll, (,$CoordArgs)
}

if (-not $SkipDeploy) {
    $deploy = Join-Path $PSScriptRoot '..\.claude\scripts\tmp-deploy-coord.ps1'
    if (Test-Path $deploy) {
        Write-Host "==> Deploying latest coordinator DLL via $deploy"
        & $deploy
    } else {
        Write-Warning "Deploy script not found at $deploy. Pass -SkipDeploy if remote is already current."
    }
}

Write-Host "==> Generating $TargetCount examples across $ShardCount shards (seed=$Seed, pool=v2, name=$Name)..."
$genOut = Invoke-Coord @('generate-corpus', "$TargetCount",
    '--seed', "$Seed", '--pool', 'v2', '--name', $Name,
    '--examples-per-shard', "$examplesPerShard")
$genOut | Write-Host

Write-Host "==> Backing up existing vocab.json -> vocab.v1.json..."
Invoke-Command -ComputerName $Coordinator -ScriptBlock {
    param($dataRoot)
    $tokenized = Join-Path $dataRoot 'corpus\tokenized'
    $src = Join-Path $tokenized 'vocab.json'
    $dst = Join-Path $tokenized 'vocab.v1.json'
    if ((Test-Path $src) -and -not (Test-Path $dst)) {
        Copy-Item $src $dst
        Write-Host "    backed up $src -> $dst"
    } elseif (Test-Path $dst) {
        Write-Host "    backup $dst already exists; preserving it"
    } else {
        Write-Host "    no prior vocab.json to back up"
    }
} -ArgumentList $DataRoot

Write-Host "==> Tokenizing $Name shards (maxVocab=5174)..."
$tokOut = Invoke-Coord @('tokenize-corpus', '5174', $Name)
$tokOut | Write-Host
$tokExit = $LASTEXITCODE
if ($tokExit -eq 3) {
    Write-Error "Tokenizer exceeded 5174 vocab cap - restoring v1 vocab."
    Invoke-Command -ComputerName $Coordinator -ScriptBlock {
        param($dataRoot)
        $tokenized = Join-Path $dataRoot 'corpus\tokenized'
        $v1 = Join-Path $tokenized 'vocab.v1.json'
        $cur = Join-Path $tokenized 'vocab.json'
        if (Test-Path $v1) { Copy-Item $v1 $cur -Force }
    } -ArgumentList $DataRoot
    exit 1
}

if (-not $SkipSeed) {
    Write-Host "==> Seeding tasks against $Name shards..."
    # seed-real-tasks does not yet honor --shard-prefix; caller can
    # run it manually after this script lands if shard targeting is
    # needed.
    Write-Warning "Automatic seeding skipped: run seed-real-tasks manually after verifying vocab."
}

Write-Host "==> Done. Monitor http://${Coordinator}:5000/admin/dashboard"
