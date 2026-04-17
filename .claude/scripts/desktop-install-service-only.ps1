param(
    [string]$WorkerApiKey,
    [string]$AdminUsername      = 'admin-d2',
    [string]$AdminPassword,
    [int]$Port = 5000
)

$ErrorActionPreference = 'Stop'

function New-Secret {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    return [Convert]::ToBase64String($bytes).Replace('+','-').Replace('/','_').TrimEnd('=')
}

if (-not $WorkerApiKey)  { $WorkerApiKey  = New-Secret }
if (-not $AdminPassword) { $AdminPassword = New-Secret }

$desktopIp = (Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    Get-NetIPAddress -AddressFamily IPv4 `
        -InterfaceAlias 'Ethernet*','Wi-Fi*' `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike '169.*' -and $_.IPAddress -ne '127.0.0.1' } |
        Select-Object -First 1 -ExpandProperty IPAddress
}).Trim()

$baseUrl = "http://${desktopIp}:$Port"

Write-Host "Installing service on PAYTON-DESKTOP:"
Write-Host "  base url        : $baseUrl"
Write-Host "  worker api key  : $WorkerApiKey"
Write-Host "  admin username  : $AdminUsername"
Write-Host "  admin password  : $AdminPassword"

$result = Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($port, $apiKey, $adminUser, $adminPass, $baseUrl)

    $ErrorActionPreference = 'Stop'
    $serviceName = 'BitNetCoordinator'

    # Ensure any prior service is gone.
    $existing = Get-Service -Name $serviceName -ErrorAction SilentlyContinue
    if ($existing -and $existing.Status -ne 'Stopped') {
        Stop-Service -Name $serviceName -Force -ErrorAction SilentlyContinue
    }
    if ($existing) {
        sc.exe delete $serviceName | Out-Null
        Start-Sleep -Milliseconds 500
    }

    $dotnet = 'C:\Program Files\dotnet\dotnet.exe'
    $dll    = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'
    if (-not (Test-Path $dll)) { throw "Missing $dll" }

    $binPath = '"' + $dotnet + '" "' + $dll + '"'

    New-Service -Name $serviceName `
        -BinaryPathName $binPath `
        -DisplayName 'BitNet Coordinator' `
        -Description 'BitNetSharp distributed CPU training coordinator (Phase D-2).' `
        -StartupType Automatic | Out-Null

    Start-Sleep -Milliseconds 800

    $regKey = "HKLM:\SYSTEM\CurrentControlSet\Services\$serviceName"
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
    New-ItemProperty -Path $regKey -Name 'Environment' -PropertyType MultiString -Value $envStrings -Force | Out-Null

    New-Item -ItemType Directory -Path 'F:\ProgramData\BitNetCoordinator' -Force | Out-Null

    if (-not (Get-NetFirewallRule -DisplayName 'BitNet Coordinator D2' -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName 'BitNet Coordinator D2' `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
            -Profile Any | Out-Null
    }

    Start-Service -Name $serviceName
    Start-Sleep -Seconds 4

    $svc = Get-Service -Name $serviceName

    [PSCustomObject]@{
        ServiceName = $serviceName
        Status      = $svc.Status.ToString()
    }
} -ArgumentList $Port, $WorkerApiKey, $AdminUsername, $AdminPassword, $baseUrl

Write-Host ''
$result | Format-List | Out-String | Write-Host

[PSCustomObject]@{
    BaseUrl       = $baseUrl
    WorkerApiKey  = $WorkerApiKey
    AdminUsername = $AdminUsername
    AdminPassword = $AdminPassword
    ServiceStatus = $result.Status
}
