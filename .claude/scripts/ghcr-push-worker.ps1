# Pushes the worker image using a scratch DOCKER_CONFIG so the
# Docker Desktop credsStore does not intercept auth. Reads the
# GHCR token from `gh auth token` which already has write:packages.
param(
    [Parameter(Mandatory=$true)]
    [string[]]$Tags
)

$ErrorActionPreference = 'Stop'

$token = (& gh auth token).Trim()
if (-not $token) { throw 'gh auth token returned empty.' }

$scratch = Join-Path $env:TEMP ("docker-cfg-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch | Out-Null

try {
    $auth = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("sharpninja:$token"))
    $config = @{
        auths = @{
            'ghcr.io' = @{ auth = $auth }
        }
    } | ConvertTo-Json -Depth 5
    Set-Content -Path (Join-Path $scratch 'config.json') -Value $config -NoNewline

    $env:DOCKER_CONFIG = $scratch
    foreach ($tag in $Tags) {
        $image = "ghcr.io/sharpninja/bitnetsharp-worker:$tag"
        Write-Host "Pushing $image" -ForegroundColor Cyan
        & docker push $image
        if ($LASTEXITCODE -ne 0) { throw "docker push $image failed ($LASTEXITCODE)" }
    }
}
finally {
    Remove-Item -Recurse -Force $scratch -ErrorAction SilentlyContinue
    Remove-Item Env:\DOCKER_CONFIG -ErrorAction SilentlyContinue
}
