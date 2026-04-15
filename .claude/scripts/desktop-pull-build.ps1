Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Continue'
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'

    # Kill any stray coordinator dotnet instances so the build
    # can overwrite the DLL.
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

    $existing = Get-Service -Name BitNetCoordinator -ErrorAction SilentlyContinue
    if ($existing -and $existing.Status -ne 'Stopped') {
        Stop-Service -Name BitNetCoordinator -Force -ErrorAction SilentlyContinue
    }
    if ($existing) {
        sc.exe delete BitNetCoordinator | Out-Null
        Start-Sleep -Milliseconds 500
    }

    $fetchOut = & git fetch origin 2>&1
    $fetchExit = $LASTEXITCODE
    $resetOut = & git reset --hard origin/main 2>&1
    $resetExit = $LASTEXITCODE
    $head = (git rev-parse --short HEAD).Trim()

    $buildOut = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE

    [PSCustomObject]@{
        FetchExit = $fetchExit
        ResetExit = $resetExit
        Head      = $head
        BuildExit = $buildExit
        BuildTail = (($buildOut | Select-Object -Last 12) -join "`n")
    }
}
