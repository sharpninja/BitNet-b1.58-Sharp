param(
    [Parameter(Mandatory=$true)]
    [string]$BundlePath
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $BundlePath)) {
    throw "Bundle not found: $BundlePath"
}

# Copy bundle to desktop via admin share
$remotePath = '\\PAYTON-DESKTOP\F$\GitHub\BitNet-b1.58-Sharp\deploy.bundle'
Copy-Item -Path $BundlePath -Destination $remotePath -Force
Write-Host "Bundle copied to PAYTON-DESKTOP"

Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Continue'
    Stop-Service -Name BitNetCoordinator -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'

    $fetch = & git fetch 'F:\GitHub\BitNet-b1.58-Sharp\deploy.bundle' main:deploy-tmp 2>&1
    $resetOut = & git reset --hard deploy-tmp 2>&1
    $null = & git branch -D deploy-tmp 2>&1
    $head = (& git rev-parse --short HEAD).Trim()

    $build = & dotnet build 'src/BitNetSharp.Distributed.Coordinator/BitNetSharp.Distributed.Coordinator.csproj' -c Release -v:minimal 2>&1
    $buildExit = $LASTEXITCODE

    Start-Service -Name BitNetCoordinator -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 4
    $svc = Get-Service -Name BitNetCoordinator

    Remove-Item 'F:\GitHub\BitNet-b1.58-Sharp\deploy.bundle' -Force -ErrorAction SilentlyContinue

    [PSCustomObject]@{
        Head = $head
        Build = $buildExit
        Status = $svc.Status.ToString()
        FetchOutput = ($fetch | Out-String).Trim()
    }
}
