Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Continue'
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'

    $restoreOut = & dotnet restore 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' 2>&1
    $restoreExit = $LASTEXITCODE

    $buildOut = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE

    [PSCustomObject]@{
        RestoreExit = $restoreExit
        RestoreTail = ($restoreOut | Select-Object -Last 15 | ForEach-Object { $_.ToString() }) -join "`n"
        BuildExit   = $buildExit
        BuildTail   = ($buildOut | Select-Object -Last 30 | ForEach-Object { $_.ToString() }) -join "`n"
    }
}
