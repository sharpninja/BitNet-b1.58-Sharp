Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $hasCqrs = Test-Path 'F:\GitHub\McpServer\src\McpServer.Cqrs\McpServer.Cqrs.csproj'
    $hasMvvm = Test-Path 'F:\GitHub\McpServer\src\McpServer.Cqrs.Mvvm\McpServer.Cqrs.Mvvm.csproj'
    $hasRepo = Test-Path 'F:\GitHub\McpServer'

    $head = ''
    if ($hasRepo) {
        Push-Location 'F:\GitHub\McpServer'
        try {
            $head = (git rev-parse --short HEAD 2>$null)
        } finally {
            Pop-Location
        }
    }

    [PSCustomObject]@{
        HasCqrs = $hasCqrs
        HasMvvm = $hasMvvm
        HasRepo = $hasRepo
        Head    = $head
    }
}
