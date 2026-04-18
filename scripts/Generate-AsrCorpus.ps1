<#
.SYNOPSIS
  Rollout wrapper for the Sonnet-backed asr-v1 ASR corpus.

.DESCRIPTION
  Chains three coordinator CLIs end-to-end on a remote Windows host
  running BitNetCoordinator:

    1. generate-asr-corpus  → asr-v1-shard-*.txt + manifest.asr-v1.json
    2. tokenize-corpus 5174 → retokenizes EVERY *-shard-*.txt in the
                              corpus dir (truckmate-v2 + asr-v1 combined)
                              so the vocab.json covers both corpora and
                              stays aligned with already-deployed weights.
    3. seed-real-tasks --shard-prefix asr-v1 → enqueues ASR work.

  Preserves the existing vocab.json as vocab.v1.json before retokenizing
  so a vocab-cap breach (exit code 3) is recoverable. Preflights the
  Coordinator__AnthropicApiKey env var on the remote because the
  generator aborts with exit 2 otherwise.

.PARAMETER Coordinator
  Remote Windows host running the BitNetCoordinator service.

.PARAMETER RepoRoot
  Path to the BitNet-b1.58-Sharp clone on the remote host.

.PARAMETER DataRoot
  Directory on the remote containing coordinator.db. Resolved from the
  service Environment registry key when omitted.

.PARAMETER TargetCount
  Total ASR-noisy examples to synthesize. Default 1000.

.PARAMETER ExamplesPerShard
  Shard split. Default 500.

.PARAMETER BatchSize
  Messages-API batch size passed to generate-asr-corpus. Default 20.

.PARAMETER Seed
  Deterministic few-shot seed. Default 42.

.PARAMETER TokensPerTask
  Forwarded to seed-real-tasks. Default 16384.

.PARAMETER MaxPerShard
  Forwarded to seed-real-tasks. Default 4.

.PARAMETER DryRun
  Forwards --dry-run to generate-asr-corpus (prints cost estimate),
  then exits. Skips tokenize and seed.

.PARAMETER SkipDeploy
  Skip the deploy-coord.ps1 step. Use when the remote already has a
  current DLL.

.PARAMETER SkipSeed
  Stop after tokenization; do not enqueue work.

.PARAMETER SkipGenerate
  Skip the generate-asr-corpus step entirely. Use when shards were
  produced out-of-band (e.g. via cline + bytedance/seed-2-0-pro or
  another operator-supplied pipeline). Requires -ShardSource.

.PARAMETER ShardSource
  Local directory containing pre-built asr-v1-shard-*.txt files.
  When -SkipGenerate is set, the script copies every matching file
  from this directory into {DataRoot}\corpus on the remote, then
  proceeds with tokenize + seed. Ignored otherwise.

.EXAMPLE
  .\Generate-AsrCorpus.ps1 -Coordinator PAYTON-DESKTOP `
      -RepoRoot 'F:\GitHub\BitNet-b1.58-Sharp' `
      -DataRoot 'F:\ProgramData\BitNetCoordinator' `
      -TargetCount 500

.EXAMPLE
  .\Generate-AsrCorpus.ps1 -Coordinator PAYTON-DESKTOP `
      -RepoRoot 'F:\GitHub\BitNet-b1.58-Sharp' `
      -DataRoot 'F:\ProgramData\BitNetCoordinator' `
      -SkipGenerate -ShardSource 'F:\tmp\asr-v1-gen'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Coordinator,

    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,

    [string]$DataRoot,

    [int]$TargetCount      = 1000,
    [int]$ExamplesPerShard = 500,
    [int]$BatchSize        = 20,
    [int]$Seed             = 42,
    [long]$TokensPerTask   = 16384,
    [int]$MaxPerShard      = 4,

    [switch]$DryRun,
    [switch]$SkipDeploy,
    [switch]$SkipSeed,
    [switch]$SkipGenerate,
    [string]$ShardSource
)

if ($SkipGenerate) {
    if (-not $ShardSource) {
        throw "-SkipGenerate requires -ShardSource pointing to a local dir with asr-v1-shard-*.txt files."
    }
    if (-not (Test-Path $ShardSource -PathType Container)) {
        throw "-ShardSource '$ShardSource' does not exist or is not a directory."
    }
    $preShards = Get-ChildItem -Path $ShardSource -Filter 'asr-v1-shard-*.txt' -File
    if ($preShards.Count -eq 0) {
        throw "-ShardSource '$ShardSource' contains no asr-v1-shard-*.txt files."
    }
}

$ErrorActionPreference = 'Stop'

if ($TargetCount -le 0 -or $ExamplesPerShard -le 0 -or $BatchSize -le 0) {
    throw "TargetCount, ExamplesPerShard, and BatchSize must be positive"
}

$remoteDll = Join-Path $RepoRoot 'src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'

function Resolve-RemoteDataRoot {
    param([string]$HostName)
    Invoke-Command -ComputerName $HostName -ScriptBlock {
        $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
        $env = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
        $dbLine = $env | Where-Object { $_ -match '^(Coordinator__DatabasePath|BITNET_COORDINATOR_DatabasePath)=' }
        if (-not $dbLine) { return $null }
        $dbPath = $dbLine -replace '^(Coordinator__DatabasePath|BITNET_COORDINATOR_DatabasePath)=', ''
        return (Split-Path -Parent $dbPath)
    }
}

function Test-RemoteAnthropicKey {
    param([string]$HostName)
    Invoke-Command -ComputerName $HostName -ScriptBlock {
        $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
        $env = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
        $line = $env | Where-Object { $_ -match '^Coordinator__AnthropicApiKey=' }
        if (-not $line) { return $false }
        $val = $line -replace '^Coordinator__AnthropicApiKey=', ''
        return -not [string]::IsNullOrWhiteSpace($val)
    }
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
        $LASTEXITCODE
    } -ArgumentList $remoteDll, (,$CoordArgs)
}

if (-not $DataRoot) {
    Write-Host "==> Resolving remote data root from service registry..."
    $DataRoot = Resolve-RemoteDataRoot -HostName $Coordinator
    if (-not $DataRoot) {
        throw "Could not resolve DataRoot from service registry on $Coordinator. Pass -DataRoot explicitly."
    }
    Write-Host "    DataRoot = $DataRoot"
}

if (-not $SkipDeploy) {
    $deploy = Join-Path $PSScriptRoot 'deploy-coord.ps1'
    if (Test-Path $deploy) {
        Write-Host "==> Deploying latest coordinator DLL via $deploy"
        & $deploy
    } else {
        Write-Warning "Deploy script not found at $deploy. Pass -SkipDeploy if remote is already current."
    }
}

if ($SkipGenerate) {
    if ($DryRun) {
        Write-Host "==> -SkipGenerate + -DryRun: nothing to price; exiting."
        return
    }
    $remoteCorpus = Join-Path $DataRoot 'corpus'
    Write-Host ("==> Copying " + $preShards.Count + " shard(s) from " + $ShardSource + " -> \\" + $Coordinator + "\" + ($remoteCorpus -replace ':','$'))
    $uncCorpus = '\\' + $Coordinator + '\' + ($remoteCorpus -replace ':','$')
    Invoke-Command -ComputerName $Coordinator -ScriptBlock {
        param($corpus)
        New-Item -ItemType Directory -Path $corpus -Force | Out-Null
    } -ArgumentList $remoteCorpus
    foreach ($f in $preShards) {
        Copy-Item -Path $f.FullName -Destination $uncCorpus -Force
        Write-Host ("    " + $f.Name + " (" + $f.Length + "B)")
    }
}
else {
    if (-not $DryRun) {
        Write-Host "==> Preflight: checking Coordinator__AnthropicApiKey on $Coordinator..."
        $hasKey = Test-RemoteAnthropicKey -HostName $Coordinator
        if (-not $hasKey) {
            throw "Coordinator__AnthropicApiKey is not set in the BitNetCoordinator service environment on $Coordinator. Set it and restart the service before rerunning. (Use -DryRun to price a run without the key.)"
        }
        Write-Host "    key present."
    }

    Write-Host "==> Generating $TargetCount ASR examples (epc=$ExamplesPerShard, batch=$BatchSize, seed=$Seed)..."
    $genArgs = @('generate-asr-corpus', "$TargetCount",
        '--seed', "$Seed",
        '--examples-per-shard', "$ExamplesPerShard",
        '--batch-size', "$BatchSize")
    if ($DryRun) { $genArgs += '--dry-run' }

    $genResult = Invoke-Coord -CoordArgs $genArgs
    $genExit = $genResult[-1]
    $genResult[0..($genResult.Count - 2)] | Write-Host
    if ($genExit -ne 0) {
        throw "generate-asr-corpus failed with exit code $genExit."
    }

    if ($DryRun) {
        Write-Host "==> Dry-run complete. No shards written, no vocab changes, no tasks seeded."
        return
    }
}

Write-Host "==> Backing up existing vocab.json -> vocab.v1.json (idempotent)..."
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

Write-Host "==> Retokenizing ALL corpus shards (truckmate-v2 + asr-v1 unified, maxVocab=5174)..."
$tokResult = Invoke-Coord -CoordArgs @('tokenize-corpus', '5174')
$tokExit = $tokResult[-1]
$tokResult[0..($tokResult.Count - 2)] | Write-Host

if ($tokExit -eq 3) {
    Write-Error "Tokenizer exceeded the 5174 vocab cap. Restoring vocab.v1.json over vocab.json; no bin shards trusted."
    Invoke-Command -ComputerName $Coordinator -ScriptBlock {
        param($dataRoot)
        $tokenized = Join-Path $dataRoot 'corpus\tokenized'
        $v1 = Join-Path $tokenized 'vocab.v1.json'
        $cur = Join-Path $tokenized 'vocab.json'
        if (Test-Path $v1) {
            Copy-Item $v1 $cur -Force
            Write-Host "    restored $v1 -> $cur"
        }
    } -ArgumentList $DataRoot
    throw "Vocab cap breach. Rerun with a smaller -TargetCount or drop newly-generated asr-v1 shards before retokenizing."
}
if ($tokExit -ne 0) {
    throw "tokenize-corpus failed with exit code $tokExit."
}

if (-not $SkipSeed) {
    Write-Host "==> Seeding tasks against asr-v1 shards (tokensPerTask=$TokensPerTask, maxPerShard=$MaxPerShard)..."
    $seedResult = Invoke-Coord -CoordArgs @('seed-real-tasks', "$TokensPerTask", "$MaxPerShard", '--shard-prefix', 'asr-v1')
    $seedExit = $seedResult[-1]
    $seedResult[0..($seedResult.Count - 2)] | Write-Host
    if ($seedExit -ne 0) {
        throw "seed-real-tasks failed with exit code $seedExit."
    }
}

Write-Host "==> Done. Monitor http://${Coordinator}:5000/admin/dashboard"
