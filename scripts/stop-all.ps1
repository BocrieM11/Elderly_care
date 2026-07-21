. "$PSScriptRoot\common.ps1"

& (Join-Path $PSScriptRoot 'stop-news.ps1')

$ports = 8016, 10095, 12393, 50000
$listeners = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in $ports }
$processIds = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($processId in $processIds) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2
$remaining = Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in $ports }
if ($remaining) {
    $remaining | Select-Object LocalPort, OwningProcess
    throw 'Some services are still running.'
}
Write-Host 'All OLV services are stopped.'
