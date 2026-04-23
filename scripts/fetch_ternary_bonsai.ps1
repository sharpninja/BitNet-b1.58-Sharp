<#
.SYNOPSIS
Downloads prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf into data/models/.

.DESCRIPTION
Pulls the 2.03 GiB Prism-Q2_0 packed Qwen3-8B GGUF from Hugging Face so it can
be fed to `dotnet run --project src/BitNetSharp.App -- import-gguf`.

Verifies:
- File size > 1.5 GiB (soft floor; actual ~2.03 GiB).
- First 4 bytes spell "GGUF" (magic).

.PARAMETER Destination
Override the default output path. Default: <repo-root>/data/models/Ternary-Bonsai-8B-Q2_0.gguf

.PARAMETER Force
Re-download even if destination already exists.

.EXAMPLE
pwsh scripts/fetch_ternary_bonsai.ps1

.EXAMPLE
pwsh scripts/fetch_ternary_bonsai.ps1 -Force
#>

[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$Url = 'https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf'
$RepoRoot = Split-Path -Parent $PSScriptRoot

if (-not $Destination) {
    $Destination = Join-Path $RepoRoot 'data/models/Ternary-Bonsai-8B-Q2_0.gguf'
}

$DestDir = Split-Path -Parent $Destination
if (-not (Test-Path $DestDir)) {
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
}

if ((Test-Path $Destination) -and -not $Force) {
    Write-Host "Already present: $Destination"
    Write-Host "Pass -Force to re-download."
} else {
    Write-Host "Downloading $Url"
    Write-Host "  -> $Destination"
    Write-Host "  (~2.03 GiB; this may take several minutes)"
    try {
        $oldPref = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $Url -OutFile $Destination -MaximumRedirection 5
    } finally {
        $ProgressPreference = $oldPref
    }
}

$item = Get-Item -LiteralPath $Destination
$sizeGiB = [math]::Round($item.Length / 1GB, 3)
Write-Host "Size: $sizeGiB GiB ($($item.Length) bytes)"

if ($item.Length -lt (1.5 * 1GB)) {
    throw "Downloaded file is smaller than 1.5 GiB ($sizeGiB GiB). Likely an error page. Re-run with -Force."
}

$fs = [System.IO.File]::Open($Destination, 'Open', 'Read')
try {
    $magic = New-Object byte[] 4
    $read = $fs.Read($magic, 0, 4)
    if ($read -ne 4) {
        throw "Failed to read GGUF magic (read $read bytes)."
    }
} finally {
    $fs.Dispose()
}

$expectedMagic = [System.Text.Encoding]::ASCII.GetBytes('GGUF')
for ($i = 0; $i -lt 4; $i++) {
    if ($magic[$i] -ne $expectedMagic[$i]) {
        $hex = ($magic | ForEach-Object { $_.ToString('X2') }) -join ' '
        throw "File magic mismatch. Expected 'GGUF', got bytes [$hex]."
    }
}

Write-Host "Magic: GGUF OK"
Write-Host "Ready. Next: dotnet run --project src/BitNetSharp.App -- import-gguf --input=`"$Destination`" --output=data/models/bonsai.bitnetsharp.gguf"
