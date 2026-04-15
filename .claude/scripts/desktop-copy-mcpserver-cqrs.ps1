# Copies the minimum McpServer files PAYTON-DESKTOP needs for the
# BitNetSharp.Distributed.Coordinator cross-repo ProjectReference
# to resolve:
#
#   F:\GitHub\McpServer\Directory.Build.props
#   F:\GitHub\McpServer\Directory.Build.targets
#   F:\GitHub\McpServer\Directory.Packages.props
#   F:\GitHub\McpServer\src\McpServer.Cqrs\**
#   F:\GitHub\McpServer\src\McpServer.Cqrs.Mvvm\**
#
# Uses a single PSSession + Copy-Item -ToSession so we do not depend
# on the target having repo credentials for the private McpServer
# repo on github.

$ErrorActionPreference = 'Stop'

$session = New-PSSession -ComputerName PAYTON-DESKTOP
try {
    Write-Host 'Creating target directories on PAYTON-DESKTOP…'
    Invoke-Command -Session $session -ScriptBlock {
        New-Item -ItemType Directory -Path 'F:\GitHub\McpServer\src' -Force | Out-Null
    }

    Write-Host 'Cleaning prior McpServer.Cqrs copies…'
    Invoke-Command -Session $session -ScriptBlock {
        Remove-Item -Recurse -Force 'F:\GitHub\McpServer\src\McpServer.Cqrs' -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force 'F:\GitHub\McpServer\src\McpServer.Cqrs.Mvvm' -ErrorAction SilentlyContinue
    }

    Write-Host 'Copying Directory.Build.props / Directory.Build.targets / Directory.Packages.props…'
    Copy-Item -ToSession $session `
        -Path 'F:\GitHub\McpServer\Directory.Build.props' `
        -Destination 'F:\GitHub\McpServer\Directory.Build.props' -Force
    Copy-Item -ToSession $session `
        -Path 'F:\GitHub\McpServer\Directory.Build.targets' `
        -Destination 'F:\GitHub\McpServer\Directory.Build.targets' -Force
    Copy-Item -ToSession $session `
        -Path 'F:\GitHub\McpServer\Directory.Packages.props' `
        -Destination 'F:\GitHub\McpServer\Directory.Packages.props' -Force

    Write-Host 'Copying McpServer.Cqrs source (excluding bin/obj)…'
    # Copy-Item -ToSession does not honor -Exclude when -Recurse is set,
    # so stage a clean tree via a temp path first. Use a staging dir on
    # the local box that only contains the files we want to ship.
    $staging = Join-Path $env:TEMP ('bitnet-cqrs-ship-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    try {
        $cqrsSrc  = 'F:\GitHub\McpServer\src\McpServer.Cqrs'
        $mvvmSrc  = 'F:\GitHub\McpServer\src\McpServer.Cqrs.Mvvm'
        $cqrsDest = Join-Path $staging 'McpServer.Cqrs'
        $mvvmDest = Join-Path $staging 'McpServer.Cqrs.Mvvm'

        robocopy $cqrsSrc $cqrsDest /MIR /XD bin obj .vs /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null
        robocopy $mvvmSrc $mvvmDest /MIR /XD bin obj .vs /NFL /NDL /NJH /NJS /NC /NS /NP | Out-Null

        Copy-Item -ToSession $session `
            -Path $cqrsDest `
            -Destination 'F:\GitHub\McpServer\src\McpServer.Cqrs' -Recurse -Force
        Copy-Item -ToSession $session `
            -Path $mvvmDest `
            -Destination 'F:\GitHub\McpServer\src\McpServer.Cqrs.Mvvm' -Recurse -Force
    } finally {
        if (Test-Path $staging) {
            Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Host 'Verifying…'
    Invoke-Command -Session $session -ScriptBlock {
        [PSCustomObject]@{
            CqrsCsproj  = Test-Path 'F:\GitHub\McpServer\src\McpServer.Cqrs\McpServer.Cqrs.csproj'
            MvvmCsproj  = Test-Path 'F:\GitHub\McpServer\src\McpServer.Cqrs.Mvvm\McpServer.Cqrs.Mvvm.csproj'
            BuildProps  = Test-Path 'F:\GitHub\McpServer\Directory.Build.props'
            Packages    = Test-Path 'F:\GitHub\McpServer\Directory.Packages.props'
        }
    }
} finally {
    Remove-PSSession $session
}
