param([switch]$NoBrowser)
. "$PSScriptRoot\common.ps1"

function Start-HiddenScript {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name
    )
    $logRoot = Join-Path $script:PortableRoot 'logs'
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    Start-Process powershell.exe `
        -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Path) `
        -RedirectStandardOutput (Join-Path $logRoot "$Name.stdout.log") `
        -RedirectStandardError (Join-Path $logRoot "$Name.stderr.log") `
        -WindowStyle Hidden | Out-Null
}

if (-not (Test-LocalPort -Port 8016)) {
    Start-HiddenScript -Path (Join-Path $PSScriptRoot 'start-companion.ps1') -Name 'companion'
}
Wait-HttpEndpoint -Uri 'http://127.0.0.1:8016/v1/langgraph/health' -TimeoutSeconds 180
Test-RemoteLlm | Out-Null
Write-Host "Companion LLM is ready: $($env:OLV_LLM_BASE_URL) ($($env:OLV_LLM_MODEL))."

& (Join-Path $PSScriptRoot 'start-news.ps1')

if (-not (Test-LocalPort -Port 10095)) {
    Start-HiddenScript -Path (Join-Path $PSScriptRoot 'start-asr.ps1') -Name 'asr'
}
if (-not (Test-LocalPort -Port 50000)) {
    Start-HiddenScript -Path (Join-Path $PSScriptRoot 'start-tts.ps1') -Name 'tts'
}
Wait-HttpEndpoint -Uri 'http://127.0.0.1:10095/health' -TimeoutSeconds 300
Wait-HttpEndpoint -Uri 'http://127.0.0.1:50000/v1/health' -TimeoutSeconds 600

if (-not (Test-LocalPort -Port 12393)) {
    Start-HiddenScript -Path (Join-Path $PSScriptRoot 'start-olv.ps1') -Name 'olv'
}
Wait-HttpEndpoint -Uri 'http://127.0.0.1:12393/' -TimeoutSeconds 180

Write-Host 'All services are ready: Companion CN 8016, News Scraper, ASR 10095, TTS 50000, OLV 12393.'
if (-not $NoBrowser) {
    Start-Process 'http://127.0.0.1:12393'
}
