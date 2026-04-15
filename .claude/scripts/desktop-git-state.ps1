Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $ErrorActionPreference = 'Stop'
    Set-Location 'F:\GitHub\BitNet-b1.58-Sharp'
    $head = git rev-parse --short HEAD 2>&1
    $status = git status --short 2>&1
    $branch = git branch --show-current 2>&1

    [PSCustomObject]@{
        Head   = $head
        Branch = $branch
        Status = $status
    }
}
