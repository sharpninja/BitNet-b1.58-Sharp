Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Continue'
    Stop-Service -Name BitNetCoordinator -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'
    $null = git fetch azure 2>&1
    $null = git reset --hard azure/main 2>&1
    $head = (git rev-parse --short HEAD).Trim()
    $build = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE
    Start-Service -Name BitNetCoordinator -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    $svc = Get-Service -Name BitNetCoordinator
    [PSCustomObject]@{ Head = $head; Build = $buildExit; Status = $svc.Status.ToString() }
}
