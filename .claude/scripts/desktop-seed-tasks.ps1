param(
    [int]$Count = 5
)

Invoke-Command -ComputerName PAYTON-DESKTOP -ScriptBlock {
    param($Count)

    $ErrorActionPreference = 'Stop'
    $binDir = 'F:\GitHub\BitNet-b1.58-Sharp\src\BitNetSharp.Distributed.Coordinator\bin\Release\net10.0'
    Set-Location $binDir

    # Load the needed assemblies. Microsoft.Data.Sqlite 10 auto-
    # initializes SQLitePCLRaw on first SqliteConnection.Open so we
    # do not need to call Batteries_V2.Init ourselves — but the
    # native e_sqlite3.dll still has to be resolvable from the
    # current directory, which is why we Set-Location above.
    $null = [System.Reflection.Assembly]::LoadFile((Join-Path $binDir 'Microsoft.Data.Sqlite.dll'))
    $null = [System.Reflection.Assembly]::LoadFile((Join-Path $binDir 'SQLitePCLRaw.core.dll'))
    $null = [System.Reflection.Assembly]::LoadFile((Join-Path $binDir 'SQLitePCLRaw.provider.e_sqlite3.dll'))
    $null = [System.Reflection.Assembly]::LoadFile((Join-Path $binDir 'SQLitePCLRaw.batteries_v2.dll'))

    # Force the provider registration so the first SqliteConnection
    # does not try to pick one automatically.
    [SQLitePCL.Batteries_V2]::Init()

    $conn = New-Object Microsoft.Data.Sqlite.SqliteConnection('Data Source=F:\ProgramData\BitNetCoordinator\coordinator.db')
    $conn.Open()

    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $insertSql = @'
INSERT INTO tasks (task_id, weight_version, shard_id, shard_offset, shard_length, tokens_per_task, k_local_steps, hp_json, state, attempt, created_at)
VALUES (@id, 1, 'shard-d2', @off, 8192, 8192, 4, '{}', 'Pending', 0, @now);
'@

    $inserted = 0
    for ($i = 1; $i -le $Count; $i++) {
        $cmd = $conn.CreateCommand()
        $cmd.CommandText = $insertSql
        [void]$cmd.Parameters.AddWithValue('@id',  "task-d2-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
        [void]$cmd.Parameters.AddWithValue('@off', $i * 8192)
        [void]$cmd.Parameters.AddWithValue('@now', $now)
        $rows = $cmd.ExecuteNonQuery()
        if ($rows -gt 0) { $inserted++ }
    }

    # Quick count validation from the same connection
    $countCmd = $conn.CreateCommand()
    $countCmd.CommandText = "SELECT state, COUNT(1) FROM tasks GROUP BY state"
    $reader = $countCmd.ExecuteReader()
    $states = @{}
    while ($reader.Read()) {
        $states[$reader.GetString(0)] = $reader.GetInt32(1)
    }
    $reader.Close()
    $conn.Close()

    [PSCustomObject]@{
        Inserted = $inserted
        States   = $states
    }
} -ArgumentList $Count
