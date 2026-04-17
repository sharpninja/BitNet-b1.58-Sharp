param(
    [int]$Count = 5,
    [long]$TokensPerTask = 262144
)

Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($Count, $TokensPerTask)

    $ErrorActionPreference = 'Continue'
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'

    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'

    # Ensure the seed-tasks CLI targets the same DB the service uses.
    # Read the current env strings from the registry so we pick up
    # whatever database path the install script set.
    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $envStrings = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
    $envMap = @{}
    foreach ($line in $envStrings) {
        $eq = $line.IndexOf('=')
        if ($eq -gt 0) {
            $envMap[$line.Substring(0, $eq)] = $line.Substring($eq + 1)
        }
    }

    $env:Coordinator__DatabasePath = $envMap['Coordinator__DatabasePath']
    $env:Coordinator__InitialWeightVersion = '1'

    $out = & 'C:\Program Files\dotnet\dotnet.exe' $dll 'seed-tasks' $Count $TokensPerTask 2>&1
    $exit = $LASTEXITCODE

    [PSCustomObject]@{
        Exit = $exit
        Output = ($out -join "`n")
    }
} -ArgumentList $Count,$TokensPerTask
