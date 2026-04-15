Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $stdoutLog = 'F:\GitHub\BitNet-b1.58-Sharp\artifacts\d2-coordinator-logs\coordinator.stdout.log'
    $stderrLog = 'F:\GitHub\BitNet-b1.58-Sharp\artifacts\d2-coordinator-logs\coordinator.stderr.log'

    $alive = Get-Process -Name BitNetSharp.Distributed.Coordinator -ErrorAction SilentlyContinue

    $stdoutTail = if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Tail 60 } else { @('<no stdout log>') }
    $stderrTail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 60 } else { @('<no stderr log>') }

    [PSCustomObject]@{
        Alive      = [bool]$alive
        Pid        = ($alive | Select-Object -First 1 -ExpandProperty Id)
        StdoutTail = ($stdoutTail -join "`n")
        StderrTail = ($stderrTail -join "`n")
    }
}
