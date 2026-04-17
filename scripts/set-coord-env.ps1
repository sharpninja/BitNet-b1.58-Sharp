param(
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [Parameter(Mandatory = $true)]
    [string]$Value
)

Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($Name, $Value)
    $regKey = 'HKLM:\SYSTEM\CurrentControlSet\Services\BitNetCoordinator'
    $env = (Get-ItemProperty $regKey -Name Environment).Environment
    $prefix = "$Name="
    $new = @()
    $found = $false
    foreach ($line in $env) {
        if ($line.StartsWith($prefix)) {
            $new += "$Name=$Value"
            $found = $true
        } else {
            $new += $line
        }
    }
    if (-not $found) { $new += "$Name=$Value" }
    Set-ItemProperty $regKey -Name Environment -Value $new -Type MultiString
    Write-Host "==> Restart service"
    Restart-Service BitNetCoordinator
    Start-Sleep -Seconds 3
    (Get-ItemProperty $regKey -Name Environment).Environment | Where-Object { $_.StartsWith($prefix) }
    Get-Service BitNetCoordinator | Format-List Name,Status
} -ArgumentList $Name, $Value
