$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\common.ps1"

$pidPath = Join-Path $script:PortableRoot 'companion_cn\data\news_scraper.pid'
if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Host 'News scraper is not running.'
    exit 0
}

$newsPid = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $newsPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $newsPid -Force
    $process.WaitForExit(5000)
}
Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
Write-Host 'News scraper is stopped.'
