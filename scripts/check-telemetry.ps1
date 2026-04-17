Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'
    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $envStrings = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
    foreach ($line in $envStrings) {
        $eq = $line.IndexOf('=')
        if ($eq -gt 0) { Set-Item -Path "env:$($line.Substring(0,$eq))" -Value $line.Substring($eq+1) }
    }
    & dotnet $dll dump-telemetry 2>&1
}
