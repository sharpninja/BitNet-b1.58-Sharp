Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $svc = Get-Service -Name 'BitNetCoordinator' -ErrorAction SilentlyContinue
    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $envVals = if (Test-Path $regKey) {
        (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
    } else { $null }

    $listen = Get-NetTCPConnection -State Listen -LocalPort 5000 -ErrorAction SilentlyContinue

    [PSCustomObject]@{
        ServicePresent = [bool]$svc
        ServiceStatus  = if ($svc) { $svc.Status.ToString() } else { '<missing>' }
        StartType      = if ($svc) { $svc.StartType.ToString() } else { '<n/a>' }
        BinPath        = if ($svc) { (Get-CimInstance Win32_Service -Filter "Name='BitNetCoordinator'").PathName } else { '<n/a>' }
        EnvVarCount    = if ($envVals) { $envVals.Count } else { 0 }
        Listener       = $listen | Select-Object LocalAddress, LocalPort, OwningProcess
    }
}
