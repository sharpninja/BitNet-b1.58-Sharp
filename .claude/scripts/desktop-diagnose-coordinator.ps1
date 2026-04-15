Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    $procs = Get-Process -Name dotnet -ErrorAction SilentlyContinue
    $matching = @()
    foreach ($p in $procs) {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($p.Id)").CommandLine
            if ($cmdLine -like '*BitNetSharp.Distributed.Coordinator.dll*') {
                $matching += [PSCustomObject]@{
                    Pid = $p.Id
                    StartTime = $p.StartTime
                    CommandLine = $cmdLine
                }
            }
        } catch { }
    }

    $listen = Get-NetTCPConnection -State Listen -LocalPort 5000 -ErrorAction SilentlyContinue

    $stdoutLog = 'F:\GitHub\BitNet-b1.58-Sharp\artifacts\d2-coordinator-logs\coordinator.stdout.log'
    $stderrLog = 'F:\GitHub\BitNet-b1.58-Sharp\artifacts\d2-coordinator-logs\coordinator.stderr.log'
    $stdoutTail = if (Test-Path $stdoutLog) { Get-Content $stdoutLog -Tail 80 } else { @() }
    $stderrTail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 40 } else { @() }

    $rules = Get-NetFirewallRule -DisplayName 'BitNet Coordinator D2' -ErrorAction SilentlyContinue

    [PSCustomObject]@{
        CoordinatorProcesses = $matching
        ListenersOnPort5000  = ($listen | Select-Object LocalAddress, LocalPort, OwningProcess)
        FirewallRule         = if ($rules) { @{ Display = $rules.DisplayName; Enabled = $rules.Enabled; Action = $rules.Action } } else { '<no rule>' }
        StdoutTail           = ($stdoutTail -join "`n")
        StderrTail           = ($stderrTail -join "`n")
    }
}
