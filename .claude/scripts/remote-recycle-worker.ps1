param(
    [Parameter(Mandatory=$true)]
    [string]$ComputerName,
    [string]$Image = 'ghcr.io/sharpninja/bitnetsharp-worker:latest',
    [string]$ContainerName = 'bitnet-worker'
)

# Pull GHCR token locally (gh CLI only on controller host) and pass to
# remote so we can write an explicit-auth scratch config and avoid
# Docker Desktop's wincred credsStore, which fails on WinRM.
$ghToken = (& gh auth token).Trim()
if (-not $ghToken) { throw 'gh auth token returned empty.' }

Invoke-Command -ComputerName $ComputerName -ArgumentList $Image,$ContainerName,$ghToken -ScriptBlock {
    param($image, $containerName, $ghToken)
    $ErrorActionPreference = 'Continue'

    # Snapshot the existing container's env so we can pass identical
    # vars to the new run — this preserves the API key, worker id,
    # coordinator URL, etc. without hard-coding them in the script.
    $envLines = @(docker inspect $containerName --format '{{range .Config.Env}}{{println .}}{{end}}' 2>$null)
    if (-not $envLines -or $envLines.Count -eq 0) {
        Write-Error "Existing container '$containerName' not found on $env:COMPUTERNAME — cannot preserve env."
        return
    }

    $wanted = @(
        'BITNET_COORDINATOR_URL','BITNET_WORKER_API_KEY','BITNET_WORKER_ID',
        'BITNET_WORKER_NAME','BITNET_HEARTBEAT_SECONDS','BITNET_LOG_LEVEL',
        'BITNET_CPU_THREADS','BITNET_HEALTH_BEACON','BITNET_SHUTDOWN_SECONDS',
        'BITNET_MODEL_PRESET','BITNET_ALLOW_SYNTHETIC_SHARDS'
    )
    $envArgs = @()
    foreach ($line in $envLines) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $name = ($line -split '=',2)[0]
        if ($wanted -contains $name) {
            $envArgs += @('-e', $line)
        }
    }

    Write-Host "Pulling $image on $env:COMPUTERNAME"
    # Use a scratch DOCKER_CONFIG so Docker Desktop's credsStore does
    # not intercept and fail over WinRM. GHCR images are public so no
    # creds are needed for a pull.
    $scratch = Join-Path $env:TEMP ("dcfg-" + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    # Write explicit inline auth for ghcr.io — docker CLI on Windows
    # still falls back to wincred when config.json has no auth entry
    # for the registry, and wincred fails over WinRM non-interactive.
    $auth = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes("sharpninja:$ghToken"))
    $cfg = '{"auths":{"ghcr.io":{"auth":"' + $auth + '"}}}'
    Set-Content -Path (Join-Path $scratch 'config.json') -Value $cfg -NoNewline
    try {
        $pull = & docker --config $scratch pull $image 2>&1
        $pull | Out-String | Write-Host
    }
    finally {
        Remove-Item -Recurse -Force $scratch -ErrorAction SilentlyContinue
    }

    Write-Host "Stopping + removing existing container"
    & docker rm -f $containerName 2>&1 | Out-String | Write-Host

    Write-Host "Starting new container"
    $runArgs = @(
        'run', '--detach',
        '--name', $containerName,
        '--restart', 'unless-stopped',
        '--cpus', '0'  # honor BITNET_CPU_THREADS inside container
    )
    # Drop --cpus flag because 0 means "unlimited"; Docker rejects literal 0.
    $runArgs = $runArgs | Where-Object { $_ -ne '--cpus' -and $_ -ne '0' }
    $runArgs += $envArgs
    $runArgs += $image

    $runOutput = & docker @runArgs 2>&1
    $runOutput | Out-String | Write-Host

    $status = & docker ps --filter "name=$containerName" --format '{{.Names}}\t{{.Image}}\t{{.Status}}' 2>&1
    [PSCustomObject]@{
        Computer = $env:COMPUTERNAME
        Status   = ($status | Out-String).Trim()
    }
}
