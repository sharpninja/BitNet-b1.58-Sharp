Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Continue'

    Stop-Service -Name BitNetCoordinator -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500

    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'
    $null = git fetch origin 2>&1
    $null = git reset --hard origin/main 2>&1
    $head = (git rev-parse --short HEAD).Trim()

    $buildOut = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE

    $dll = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0\BitNetSharp.Distributed.Coordinator.dll'
    $env:Coordinator__DatabasePath = 'F:\ProgramData\BitNetCoordinator\coordinator.db'

    $tokOut = & 'C:\Program Files\dotnet\dotnet.exe' $dll 'tokenize-corpus' 8000 2>&1
    $tokExit = $LASTEXITCODE

    Start-Service -Name BitNetCoordinator -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    $svc = Get-Service -Name BitNetCoordinator

    [PSCustomObject]@{
        Head     = $head
        Build    = $buildExit
        TokExit  = $tokExit
        TokOut   = ($tokOut -join "`n")
        Service  = if ($svc) { $svc.Status.ToString() } else { 'missing' }
    }
}
