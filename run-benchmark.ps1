<#
.SYNOPSIS
    Runs the full benchmark suite locally and publishes the report to the gh-pages branch.

.PARAMETER PerplexitySamplePercent
    Percentage (0-100] of the WikiText2 validation set to use for perplexity evaluation.
    Default: 100 (full corpus).

.PARAMETER SkipRestore
    Skip dotnet restore (use when dependencies are already up to date).

.PARAMETER SkipBuild
    Skip dotnet build (use when binaries are already current).

.PARAMETER SkipTest
    Skip SlowLane tests (use when only re-generating the report).

.PARAMETER SkipDeploy
    Generate the report but do not push to gh-pages.

.PARAMETER GhPagesRemote
    Git remote to push gh-pages to. Default: origin (GitHub).

.EXAMPLE
    .\run-benchmark.ps1

.EXAMPLE
    .\run-benchmark.ps1 -PerplexitySamplePercent 10 -SkipRestore -SkipBuild
#>
param(
    [ValidateRange(1, 100)]
    [int]$PerplexitySamplePercent = 100,

    [switch]$SkipRestore,
    [switch]$SkipBuild,
    [switch]$SkipTest,
    [switch]$SkipDeploy,

    [string]$GhPagesRemote = 'origin'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot      = $PSScriptRoot
$SolutionFile  = Join-Path $RepoRoot 'BitNet-b1.58-Sharp.slnx'
$AppProject    = Join-Path $RepoRoot 'src\BitNetSharp.App\BitNetSharp.App.csproj'
$ArtifactsDir  = Join-Path $RepoRoot 'artifacts\benchmark-report'
$WorktreeDir   = Join-Path $RepoRoot '.gh-pages-worktree'

function Step([string]$Name, [scriptblock]$Block) {
    Write-Host ""
    Write-Host "--- $Name ---" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "$Name failed (exit $LASTEXITCODE)"
    }
}

Push-Location $RepoRoot
try {
    $Commit = git rev-parse HEAD
    Write-Host "Repo:   $RepoRoot"
    Write-Host "Commit: $Commit"
    Write-Host "Perplexity sample: $PerplexitySamplePercent%"

    # ── 1. Restore ────────────────────────────────────────────────────────────
    if (-not $SkipRestore) {
        Step 'Restore' {
            dotnet restore $SolutionFile
        }
    }

    # ── 2. Build ──────────────────────────────────────────────────────────────
    if (-not $SkipBuild) {
        Step 'Build' {
            dotnet build $SolutionFile --configuration Release --no-restore
        }
    }

    # ── 3. Test (SlowLane) ────────────────────────────────────────────────────
    if (-not $SkipTest) {
        Step 'Test (Category=SlowLane)' {
            dotnet test $SolutionFile `
                --configuration Release `
                --no-build --no-restore `
                --filter 'Category=SlowLane'
        }
    }

    # ── 4. Generate report ────────────────────────────────────────────────────
    Step 'Generate benchmark report' {
        if (Test-Path $ArtifactsDir) { Remove-Item $ArtifactsDir -Recurse -Force }
        New-Item -ItemType Directory -Path $ArtifactsDir -Force | Out-Null

        dotnet run `
            --framework net10.0 `
            --configuration Release `
            --project $AppProject `
            -- `
            benchmark-report `
            --model=bitnet-b1.58-sharp `
            --compare-model=traditional-local `
            "--commit=$Commit" `
            "--output=$ArtifactsDir" `
            "--perplexity-sample-percent=$PerplexitySamplePercent"
    }

    if (-not (Test-Path (Join-Path $ArtifactsDir 'index.html'))) {
        throw "Report generation failed: index.html not found in $ArtifactsDir"
    }
    Write-Host "Report generated: $ArtifactsDir" -ForegroundColor Green

    # ── 5. Push to gh-pages ───────────────────────────────────────────────────
    if (-not $SkipDeploy) {
        Step 'Publish to gh-pages' {
            # Clean up any leftover worktree from a previous failed run
            if (Test-Path $WorktreeDir) {
                git worktree remove $WorktreeDir --force 2>$null
                Remove-Item $WorktreeDir -Recurse -Force -ErrorAction SilentlyContinue
            }

            # Create or fetch gh-pages branch
            $branchExists = git ls-remote --heads $GhPagesRemote gh-pages | Select-String 'gh-pages'
            if ($branchExists) {
                git fetch $GhPagesRemote gh-pages
                git worktree add $WorktreeDir gh-pages 2>$null
                if ($LASTEXITCODE -ne 0) {
                    git worktree add $WorktreeDir --track -b gh-pages "$GhPagesRemote/gh-pages"
                }
            } else {
                # First time: create orphan gh-pages branch
                git worktree add --orphan -b gh-pages $WorktreeDir
            }

            try {
                Push-Location $WorktreeDir

                # Clear existing content (keep .git)
                Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' } |
                    Remove-Item -Recurse -Force

                # Copy report
                Copy-Item -Path "$ArtifactsDir\*" -Destination $WorktreeDir -Recurse -Force

                git add -A
                $msg = "Benchmark report $($Commit.Substring(0,8)) $(Get-Date -Format 'yyyy-MM-dd HH:mm') - perplexity $PerplexitySamplePercent%"
                git commit -m $msg
                git push $GhPagesRemote gh-pages

                Write-Host "Published to $GhPagesRemote/gh-pages" -ForegroundColor Green
            } finally {
                Pop-Location
                git worktree remove $WorktreeDir --force 2>$null
            }
        }
    }

    Write-Host ""
    Write-Host "Done." -ForegroundColor Green

} finally {
    Pop-Location
}
