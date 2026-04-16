param(
    [string]$CorpusDir = 'C:\Users\kingd\AppData\Local\Temp\train-prep\corpus',
    [string]$OutputFile = 'C:\Users\kingd\AppData\Local\Temp\train-prep\corpus\truckmate-training.jsonl'
)

$ErrorActionPreference = 'Stop'

$writer = [System.IO.StreamWriter]::new($OutputFile, $false, [System.Text.Encoding]::UTF8)
$count = 0

Get-ChildItem $CorpusDir -Filter '*.txt' | Sort-Object Name | ForEach-Object {
    foreach ($line in [System.IO.File]::ReadLines($_.FullName)) {
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $idx = $line.IndexOf('[INTENT]')
        if ($idx -lt 0) { continue }
        $prompt = $line.Substring(0, $idx).Replace('[USER]', '').Trim()
        $response = $line.Substring($idx + 8).Trim()
        # Manual JSON construction avoids ConvertTo-Json per-line overhead
        $escapedPrompt = $prompt.Replace('\', '\\').Replace('"', '\"')
        $escapedResponse = $response.Replace('\', '\\').Replace('"', '\"')
        $writer.WriteLine("{`"prompt`":`"$escapedPrompt`",`"response`":`"$escapedResponse`"}")
        $count++
    }
}

$writer.Close()
Write-Host "Wrote $count examples to $OutputFile"
