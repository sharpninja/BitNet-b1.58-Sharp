# Installs BitNetSharp.Distributed.Coordinator as a Windows service
# on PAYTON-DESKTOP. Idempotent — safely stops + removes any prior
# service instance before re-creating. Generates fresh admin and
# worker secrets, stores them in the service's Environment registry
# key (REG_MULTI_SZ), and starts the service.
#
# Pass -NoStart to install + populate env but leave the service in
# Stopped state (manual start via sc.exe start BitNetCoordinator).
#
# Requires elevated privileges on PAYTON-DESKTOP (sc.exe + registry
# writes under HKLM\SYSTEM\CurrentControlSet\Services). The
# remoteadmin user on PAYTON-DESKTOP has admin rights by design.

param(
    [switch]$NoStart,
    [switch]$SkipGitPull
)

$ErrorActionPreference = 'Stop'

function New-Secret {
    $bytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    return [Convert]::ToBase64String($bytes).Replace('+','-').Replace('/','_').TrimEnd('=')
}

$serviceName        = 'BitNetCoordinator'
$coordinatorPort    = 5000

# Resolve the routable IPv4. Windows prefers IPv6 first which does
# not route cleanly over the LAN; pin to IPv4.
$desktopIp = (Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    Get-NetIPAddress -AddressFamily IPv4 `
        -InterfaceAlias 'Ethernet*','Wi-Fi*' `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike '169.*' -and $_.IPAddress -ne '127.0.0.1' } |
        Select-Object -First 1 -ExpandProperty IPAddress
}).Trim()

if (-not $desktopIp) {
    throw 'Could not resolve PAYTON-DESKTOP LAN IPv4 address.'
}

$baseUrl       = "http://${desktopIp}:$coordinatorPort"
$workerApiKey  = New-Secret
$adminUsername = 'admin-d2'
$adminPassword = New-Secret

Write-Host "Installing $serviceName on PAYTON-DESKTOP bound to $baseUrl"
Write-Host "  worker api key : $workerApiKey"
Write-Host "  admin username : $adminUsername"
Write-Host "  admin password : $adminPassword"

$remote = Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param(
        $serviceName, $port, $apiKey,
        $adminUser, $adminPass, $baseUrl, $skipStart, $skipPull)

    $ErrorActionPreference = 'Stop'

    # ── 1. Kill any stray ad-hoc coordinator processes FIRST so
    #      the build in step 3 can overwrite the DLL. cmd.exe wraps
    #      (launch.cmd) and the dotnet.exe child both need to go.
    Get-Process -Name dotnet -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
            if ($cmdLine -like '*BitNetSharp.Distributed.Coordinator.dll*') {
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            }
        } catch { }
    }
    Get-Process -Name cmd -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
            if ($cmdLine -like '*launch.cmd*') {
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            }
        } catch { }
    }

    # ── 2. Stop + delete any prior service instance so sc.exe
    #      create does not collide, AND so an already-installed
    #      service is not holding the DLL while we rebuild.
    $existing = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($existing -and $existing.Status -ne 'Stopped') {
        try { Stop-Service -Name $serviceName -Force -ErrorAction Stop } catch {}
    }
    if ($existing) {
        & sc.exe delete $serviceName | Out-Null
        Start-Sleep -Milliseconds 500
    }

    # ── 3. Pull latest source + rebuild Release ──────────────────
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'
    if (-not $skipPull) {
        $pull = (git fetch origin 2>&1) + "`n" + (git reset --hard origin/main 2>&1)
    }
    $head = (git rev-parse --short HEAD).Trim()

    $buildOut = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE
    if ($buildExit -ne 0) {
        throw "Build failed (exit $buildExit): $(($buildOut | Select-Object -Last 20) -join "`n")"
    }

    $dotnet = 'C:\Program Files\dotnet\dotnet.exe'
    $dll    = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'
    if (-not (Test-Path $dll)) { throw "Missing $dll" }

    # ── 4. Create the service via New-Service which handles the
    #      binary path quoting correctly (sc.exe would need a
    #      --% stop-parser dance that is awkward inside an
    #      Invoke-Command scriptblock).
    $binPath = '"' + $dotnet + '" "' + $dll + '"'

    New-Service -Name $serviceName `
        -BinaryPathName $binPath `
        -DisplayName 'BitNet Coordinator' `
        -Description 'BitNetSharp distributed CPU training coordinator (Phase D-2).' `
        -StartupType Automatic | Out-Null

    & sc.exe failure $serviceName reset= 86400 actions= restart/5000/restart/10000/restart/30000 | Out-Null

    # Give the HKLM: PS drive a beat to notice the new service key.
    Start-Sleep -Milliseconds 800

    # ── 5. Populate service Environment via registry ────────────
    $envStrings = @(
        "ASPNETCORE_ENVIRONMENT=Production"
        "ASPNETCORE_URLS=http://0.0.0.0:$port"
        "DOTNET_gcServer=1"
        "Coordinator__BaseUrl=$baseUrl"
        "Coordinator__DatabasePath=F:\ProgramData\BitNetCoordinator\coordinator.db"
        "Coordinator__HeartbeatIntervalSeconds=15"
        "Coordinator__StaleWorkerThresholdSeconds=120"
        "Coordinator__TargetTaskDurationSeconds=60"
        "Coordinator__FullStepEfficiency=0.25"
        "Coordinator__Admin__Username=$adminUser"
        "Coordinator__Admin__Password=$adminPass"
        "Coordinator__WorkerApiKey=$apiKey"
    )
    $regKey = "HKLM:\SYSTEM\CurrentControlSet\Services\$serviceName"
    # New-ItemProperty on a fresh service key sometimes fails with
    # a transient "path not found" — retry for up to 5 seconds.
    $regDeadline = (Get-Date).AddSeconds(5)
    while ((Get-Date) -lt $regDeadline -and -not (Test-Path $regKey)) {
        Start-Sleep -Milliseconds 200
    }
    if (-not (Test-Path $regKey)) {
        throw "Service registry key $regKey not present after sc.exe create"
    }
    New-ItemProperty -Path $regKey -Name 'Environment' -PropertyType MultiString -Value $envStrings -Force | Out-Null

    # Ensure the data directory exists so the service can write its
    # SQLite file on first startup.
    New-Item -ItemType Directory -Path 'F:\ProgramData\BitNetCoordinator' -Force | Out-Null

    # ── 6. Firewall rule for port 5000 ──────────────────────────
    try {
        if (-not (Get-NetFirewallRule -DisplayName 'BitNet Coordinator D2' -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName 'BitNet Coordinator D2' `
                -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
                -Profile Any | Out-Null
        }
    } catch { Write-Warning "firewall: $_" }

    # ── 7. Start the service (skipped when -NoStart) ────────────
    if (-not $skipStart) {
        & sc.exe start $serviceName | Out-Null
        Start-Sleep -Seconds 4
    }

    $svc = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    $listen = if ($skipStart) { $null } else {
        Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue |
            Select-Object -First 1 LocalAddress, LocalPort, OwningProcess
    }

    [PSCustomObject]@{
        GitHead     = $head
        BuildExit   = $buildExit
        ServiceName = $serviceName
        Status      = if ($svc) { $svc.Status.ToString() } else { 'missing' }
        Listener    = $listen
    }
} -ArgumentList $serviceName, $coordinatorPort, $workerApiKey, $adminUsername, $adminPassword, $baseUrl, $NoStart.IsPresent, $SkipGitPull.IsPresent

Write-Host ''
Write-Host '── Service install result ────────────────────────────────────'
$remote | Format-List | Out-String | Write-Host

[PSCustomObject]@{
    ServiceName   = $serviceName
    BaseUrl       = $baseUrl
    WorkerApiKey  = $workerApiKey
    AdminUsername = $adminUsername
    AdminPassword = $adminPassword
    ServiceStatus = $remote.Status
}
