Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Stop'

    if (-not (Test-Path 'F:\GitHub')) {
        New-Item -ItemType Directory -Path 'F:\GitHub' -Force | Out-Null
    }

    Set-Location 'F:\GitHub'

    if (-not (Test-Path 'F:\GitHub\BitNet-b1.58-Sharp')) {
        Write-Host 'Cloning BitNet-b1.58-Sharp from github…'
        git clone https://github.com/sharpninja/BitNet-b1.58-Sharp.git 2>&1 | Write-Host
    }

    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'
    Write-Host 'Fetching latest…'
    git fetch origin 2>&1 | Write-Host
    Write-Host 'Resetting to origin/main…'
    git reset --hard origin/main 2>&1 | Write-Host
    $head = (git rev-parse --short HEAD).Trim()
    [PSCustomObject]@{ Head = $head }
}
