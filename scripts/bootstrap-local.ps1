$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\common.ps1"

$python = Resolve-VoicePython
$null = Resolve-ModelPath -Override $env:OLV_FUNASR_MODEL `
    -RelativePath 'models\Fun-ASR-Nano-2512' `
    -DisplayName 'Fun-ASR model'
$fishModel = Resolve-ModelPath -Override $env:OLV_FISH_MODEL `
    -RelativePath 'models\fish-speech-1.5' `
    -DisplayName 'Fish Speech 1.5 model'

Write-Host "Checking voice environment: $python"
& $python -c "import torch, funasr, gradio, pkg_resources, sherpa_onnx; assert torch.cuda.is_available(); print('Voice environment OK:', torch.__version__, funasr.__version__, gradio.__version__, sherpa_onnx.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Voice environment validation failed.' }

$fishPython = $env:OLV_FISH_PYTHON
if (-not $fishPython -or -not (Test-Path -LiteralPath $fishPython)) {
    throw "Fish Speech Python was not found: $fishPython"
}
$fishSource = $env:OLV_FISH_SOURCE
if (-not $fishSource -or -not (Test-Path -LiteralPath $fishSource)) {
    throw "Fish Speech source was not found: $fishSource"
}
$env:PYTHONPATH = "$(Join-Path $script:PortableRoot 'services\fishspeech');$fishSource"
Write-Host "Checking Fish Speech environment: $fishPython"
& $fishPython -c "import torch, fastapi, uvicorn; assert torch.cuda.is_available(); print('Fish Speech environment OK:', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Fish Speech environment validation failed.' }
Write-Host "Fish Speech model OK: $fishModel"

Test-RemoteLlm | Out-Null
Write-Host "Remote LLM OK: $($env:OLV_LLM_BASE_URL) ($($env:OLV_LLM_MODEL))"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is not installed or not available in PATH.'
}

$appRoot = Join-Path $script:PortableRoot 'app'
Push-Location $appRoot
try {
    & uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
}
finally {
    Pop-Location
}

Write-Host 'Local bootstrap completed.'
