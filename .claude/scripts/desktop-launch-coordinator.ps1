# Launches the BitNetSharp coordinator on PAYTON-DESKTOP bound to
# http://0.0.0.0:5000 and FULLY DETACHED from the PS remoting
# session. Writes a launch.cmd batch file on the remote that sets
# env vars and invokes `dotnet`, then uses
# Invoke-CimMethod Win32_Process.Create to spawn it — that API
# creates a new process whose parent is WMI, not the PS session,
# so the coordinator survives after Invoke-Command returns.
#
# Returns the PID, log paths, and the generated admin + worker
# credentials so the caller can start the worker half of the D-2
# proof with matching secrets.

$ErrorActionPreference = 'Stop'

function New-Secret {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    return [Convert]::ToBase64String($bytes).Replace('+','-').Replace('/','_').TrimEnd('=')
}

$workerApiKey    = New-Secret
$adminUsername   = 'admin-d2'
$adminPassword   = New-Secret
$coordinatorPort = 5000
$baseUrl         = "http://PAYTON-DESKTOP:$coordinatorPort"

Write-Host "Launching coordinator on PAYTON-DESKTOP bound to $baseUrl"
Write-Host "  worker api key : $workerApiKey"
Write-Host "  admin username : $adminUsername"
Write-Host "  admin password : $adminPassword"

$remote = Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($port, $apiKey, $adminUser, $adminPass, $baseUrl)

    $ErrorActionPreference = 'Stop'

    # Kill any prior coordinator processes.
    Get-Process -Name BitNetSharp.Distributed.Coordinator -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Get-Process -Name dotnet -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path -like '*dotnet*' } |
        ForEach-Object {
            try {
                $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine
                if ($cmdLine -like '*BitNetSharp.Distributed.Coordinator.dll*') {
                    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
                }
            } catch { }
        }

    # Best-effort firewall rule.
    try {
        if (-not (Get-NetFirewallRule -DisplayName 'BitNet Coordinator D2' -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName 'BitNet Coordinator D2' `
                -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
                -Profile Any | Out-Null
        }
    } catch { Write-Warning "firewall: $_" }

    $logDir  = 'F:\GitHub\BitNet-b1.58-Sharp\artifacts\d2-coordinator-logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $stdoutLog = Join-Path $logDir 'coordinator.stdout.log'
    $stderrLog = Join-Path $logDir 'coordinator.stderr.log'
    $launchCmd = Join-Path $logDir 'launch.cmd'
    $dbPath    = Join-Path $logDir 'coordinator.db'

    Set-Content -Path $stdoutLog -Value '' -Force
    Set-Content -Path $stderrLog -Value '' -Force

    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'
    if (-not (Test-Path $dll)) {
        throw "Coordinator assembly not found at $dll"
    }

    # Build launch.cmd that sets env vars and invokes dotnet. cmd.exe
    # is the one redirecting stdout/stderr to files; when WMI spawns
    # the batch file the dotnet child inherits those redirections.
    $cmdLines = @(
        '@echo off'
        'set ASPNETCORE_ENVIRONMENT=Development'
        "set ASPNETCORE_URLS=http://0.0.0.0:$port"
        "set Coordinator__BaseUrl=$baseUrl"
        "set Coordinator__DatabasePath=$dbPath"
        'set Coordinator__HeartbeatIntervalSeconds=15'
        'set Coordinator__StaleWorkerThresholdSeconds=120'
        'set Coordinator__TargetTaskDurationSeconds=60'
        'set Coordinator__FullStepEfficiency=0.25'
        "set Coordinator__Admin__Username=$adminUser"
        "set Coordinator__Admin__Password=$adminPass"
        "set Coordinator__WorkerApiKey=$apiKey"
        "cd /d F:\GitHub\BitNet-b1.58-Sharp"
        "dotnet `"$dll`" 1>> `"$stdoutLog`" 2>> `"$stderrLog`""
    )
    Set-Content -Path $launchCmd -Value $cmdLines -Encoding ASCII

    # WMI Win32_Process.Create spawns outside the PS remoting session
    # tree so the process survives after this Invoke-Command returns.
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
        -Arguments @{
            CommandLine      = "cmd.exe /c `"$launchCmd`""
            CurrentDirectory = 'F:\GitHub\BitNet-b1.58-Sharp'
        }

    if ($result.ReturnValue -ne 0) {
        throw "Win32_Process.Create returned $($result.ReturnValue)"
    }

    [PSCustomObject]@{
        Pid       = $result.ProcessId
        StdoutLog = $stdoutLog
        StderrLog = $stderrLog
        LaunchCmd = $launchCmd
        DbPath    = $dbPath
    }
} -ArgumentList $coordinatorPort, $workerApiKey, $adminUsername, $adminPassword, $baseUrl

Write-Host ''
Write-Host '── Remote coordinator spawned ──────────────────────────────────'
$remote | Format-List | Out-String | Write-Host

[PSCustomObject]@{
    WorkerApiKey  = $workerApiKey
    AdminUsername = $adminUsername
    AdminPassword = $adminPassword
    BaseUrl       = $baseUrl
    RemotePid     = $remote.Pid
    StdoutLog     = $remote.StdoutLog
    StderrLog     = $remote.StderrLog
}
