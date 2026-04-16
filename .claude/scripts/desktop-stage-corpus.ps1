param([int]$Count = 50000)

$ErrorActionPreference = 'Stop'

# 1. Pull + rebuild on PAYTON-DESKTOP
Write-Host "Pulling latest and rebuilding on PAYTON-DESKTOP…"
Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($Count)
    $ErrorActionPreference = 'Continue'
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'

    # Stop service first so the DLL isn't locked during build.
    Stop-Service -Name BitNetCoordinator -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    $null = git fetch origin 2>&1
    $null = git reset --hard origin/main 2>&1
    $head = (git rev-parse --short HEAD).Trim()

    $buildOut = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE

    if ($buildExit -ne 0) {
        throw "Build failed (exit $buildExit)"
    }

    # 2. Generate corpus
    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'

    # Read the service's database path from registry so the corpus
    # lands in the same parent directory the running service uses.
    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $envStrings = (Get-ItemProperty -Path $regKey -Name Environment -ErrorAction SilentlyContinue).Environment
    $dbPath = 'F:\ProgramData\BitNetCoordinator\coordinator.db'
    foreach ($line in $envStrings) {
        if ($line -like 'Coordinator__DatabasePath=*') {
            $dbPath = $line.Split('=', 2)[1]
        }
    }

    $env:Coordinator__DatabasePath = $dbPath
    $genOut = & 'C:\Program Files\dotnet\dotnet.exe' $dll 'generate-corpus' $Count 2>&1
    $genExit = $LASTEXITCODE

    # 3. Restart the service so it picks up the new corpus directory
    Restart-Service -Name BitNetCoordinator -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    $svc = Get-Service -Name BitNetCoordinator -ErrorAction SilentlyContinue

    [PSCustomObject]@{
        Head     = $head
        Build    = $buildExit
        GenExit  = $genExit
        GenOut   = ($genOut -join "`n")
        Service  = if ($svc) { $svc.Status.ToString() } else { 'missing' }
    }
} -ArgumentList $Count

Write-Host 'Done.'
