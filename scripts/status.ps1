$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\common.ps1"

$ports = [ordered]@{
    'Companion CN' = 8016
    'Fun-ASR' = 10095
    'Open-LLM-VTuber' = 12393
    'Fish Speech TTS' = 50000
}

$rows = foreach ($entry in $ports.GetEnumerator()) {
    $listener = Get-NetTCPConnection -LocalPort $entry.Value -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    [PSCustomObject]@{
        Service = $entry.Key
        Port = $entry.Value
        Running = $null -ne $listener
        ProcessId = if ($listener) { $listener.OwningProcess } else { $null }
    }
}
$rows | Format-Table -AutoSize

$newsPidPath = Join-Path $script:PortableRoot 'companion_cn\data\news_scraper.pid'
$newsPid = $null
$newsRunning = $false
if (Test-Path -LiteralPath $newsPidPath) {
    $newsPid = [int](Get-Content -LiteralPath $newsPidPath -Raw)
    $newsRunning = $null -ne (Get-Process -Id $newsPid -ErrorAction SilentlyContinue)
}
$newsDbPath = Join-Path $script:PortableRoot 'companion_cn\data\news.db'
$newsDb = Get-Item -LiteralPath $newsDbPath -ErrorAction SilentlyContinue
[PSCustomObject]@{
    Service = 'News Scraper'
    Running = $newsRunning
    ProcessId = if ($newsRunning) { $newsPid } else { $null }
    DatabaseBytes = if ($newsDb) { $newsDb.Length } else { 0 }
    DatabaseUpdated = if ($newsDb) { $newsDb.LastWriteTime } else { $null }
} | Format-Table -AutoSize

try {
    Test-RemoteLlm | Out-Null
    Write-Host "Companion LLM: ready ($($env:OLV_LLM_BASE_URL), model $($env:OLV_LLM_MODEL))"
}
catch {
    Write-Host "Companion LLM: unavailable ($($_.Exception.Message))"
}
