Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'
    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $envStrings = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
    foreach ($line in $envStrings) {
        $eq = $line.IndexOf('=')
        if ($eq -gt 0) { Set-Item -Path "env:$($line.Substring(0,$eq))" -Value $line.Substring($eq+1) }
    }
    Write-Host '==> Stopping coordinator service'
    Stop-Service BitNetCoordinator
    Start-Sleep -Seconds 3
    Write-Host '==> Purge pending'
    & dotnet $dll purge-pending 2>&1
    Write-Host '==> Reseed v1 (K=1)'
    & dotnet $dll seed-real-tasks 16384 2 --shard-prefix truckmate-v1 2>&1 | Select-Object -Last 6
    Write-Host '==> Reseed v2 (K=1)'
    & dotnet $dll seed-real-tasks 16384 2 --shard-prefix truckmate-v2 2>&1 | Select-Object -Last 6
    Write-Host '==> Start coordinator service'
    Start-Service BitNetCoordinator
    Start-Sleep -Seconds 3
    Get-Service BitNetCoordinator | Format-List Name,Status
}
