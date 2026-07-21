. "$PSScriptRoot\common.ps1"

if (Test-LocalPort -Port 8016) {
    Write-Host 'Companion CN is already running on port 8016.'
    exit 0
}

$python = Join-Path $script:PortableRoot 'companion_cn\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Companion CN Python was not found. Recreate it with: uv venv companion_cn\.venv --clear'
}

Push-Location $script:PortableRoot
try {
    & $python -m companion_cn.main
    if ($LASTEXITCODE -ne 0) { throw "Companion CN exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
