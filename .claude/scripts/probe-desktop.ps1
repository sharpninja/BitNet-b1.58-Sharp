Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    $git    = Get-Command git    -ErrorAction SilentlyContinue
    $ng     = Get-Command ngrok  -ErrorAction SilentlyContinue
    $repo   = Test-Path 'F:\GitHub\BitNet-b1.58-Sharp'

    $head = ''
    if ($repo) {
        Push-Location 'F:\GitHub\BitNet-b1.58-Sharp'
        try {
            $head = (git rev-parse --short HEAD 2>$null)
        } finally {
            Pop-Location
        }
    }

    [PSCustomObject]@{
        DotNetPath     = if ($dotnet) { $dotnet.Source } else { '<missing>' }
        DotNetSdks     = if ($dotnet) { (dotnet --list-sdks 2>$null) -join '; ' } else { '<none>' }
        DotNetRuntimes = if ($dotnet) { (dotnet --list-runtimes 2>$null) -join '; ' } else { '<none>' }
        GitPath        = if ($git)    { $git.Source } else { '<missing>' }
        NgrokPath      = if ($ng)     { $ng.Source  } else { '<missing>' }
        RepoExists     = $repo
        RepoHead       = $head
    }
}
