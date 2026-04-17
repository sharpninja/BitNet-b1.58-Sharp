param([int]$Limit = 40, [string]$MinLevel = '')

Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($Limit, $MinLevel)
    $ErrorActionPreference = 'Continue'
    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'

    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $envStrings = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
    foreach ($line in $envStrings) {
        $eq = $line.IndexOf('=')
        if ($eq -gt 0) {
            $name = $line.Substring(0, $eq)
            $value = $line.Substring($eq + 1)
            Set-Item -Path "env:$name" -Value $value
        }
    }

    $cmdArgs = @('dump-events', $Limit)
    if ($MinLevel) { $cmdArgs += $MinLevel }
    & 'C:\Program Files\dotnet\dotnet.exe' $dll @cmdArgs 2>&1
} -ArgumentList $Limit, $MinLevel
