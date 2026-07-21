$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\common.ps1"

$logRoot = Join-Path $script:PortableRoot 'logs'
$dataRoot = Join-Path $script:PortableRoot 'companion_cn\data'
$pidPath = Join-Path $dataRoot 'news_scraper.pid'
$python = Join-Path $script:PortableRoot 'companion_cn\.venv\Scripts\python.exe'

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path $dataRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Companion CN Python was not found. Run scripts\install.ps1 first.'
}

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "News scraper is already running (PID $existingPid)."
        exit 0
    }
    Remove-Item -LiteralPath $pidPath -Force
}

$process = Start-Process -FilePath $python `
    -ArgumentList @('-m', 'companion_cn.news_scraper', '--daemon', '--interval', '5') `
    -WorkingDirectory $script:PortableRoot `
    -RedirectStandardOutput (Join-Path $logRoot 'news.stdout.log') `
    -RedirectStandardError (Join-Path $logRoot 'news.stderr.log') `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Start-Sleep -Milliseconds 500
if (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    throw 'News scraper exited during startup. Check logs\news.stderr.log.'
}

Write-Host "News scraper started (PID $($process.Id)); it refreshes every 5 hours."
