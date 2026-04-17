$src  = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0'
$dest = '\\PAYTON-DESKTOP\F$\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0'

Write-Host "Stopping remote service..."
Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock { Stop-Service BitNetCoordinator -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

Write-Host "Mirroring $src -> $dest"
$rc = robocopy $src $dest /MIR /NFL /NDL /NJH /NP /R:2 /W:2
Write-Host "robocopy exit=$LASTEXITCODE"

Write-Host "Starting remote service..."
Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock { Start-Service BitNetCoordinator }
Start-Sleep -Seconds 5

Write-Host "Remote service state + DLL timestamps:"
Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    (Get-Service BitNetCoordinator).Status
    Get-ChildItem 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0' -Filter 'BitNetSharp.*.dll' | Select-Object Name, Length, LastWriteTime
}
