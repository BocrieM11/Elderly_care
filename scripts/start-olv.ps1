. "$PSScriptRoot\common.ps1"

if (Test-LocalPort -Port 12393) {
    Write-Host 'Open-LLM-VTuber is already running on port 12393.'
    exit 0
}
if (-not $env:OLV_BACKEND_PYTHON -and -not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not installed or not available in PATH.'
}

$appRoot = Join-Path $script:PortableRoot 'app'
Push-Location $appRoot
try {
    if ($env:OLV_BACKEND_PYTHON) {
        & $env:OLV_BACKEND_PYTHON run_server.py
    }
    else {
        & uv run --frozen run_server.py
    }
    if ($LASTEXITCODE -ne 0) { throw "Open-LLM-VTuber exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
