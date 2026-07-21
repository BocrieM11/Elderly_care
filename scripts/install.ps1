. "$PSScriptRoot\common.ps1"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Install Ollama for Windows before continuing.'
}

$runtimeRoot = Join-Path $script:PortableRoot 'runtime'
$voiceEnv = Join-Path $runtimeRoot 'voice-env'
$voicePython = Join-Path $voiceEnv 'Scripts\python.exe'
$requirements = Join-Path $script:PortableRoot 'services\requirements-voice.txt'

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

if (-not (Test-Path -LiteralPath $voicePython)) {
    Write-Host 'Creating the shared ASR/TTS Python 3.10 environment...'
    & uv venv $voiceEnv --python 3.10
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the voice environment.' }
}

Write-Host 'Installing CUDA 12.8 PyTorch for ASR/TTS...'
& uv pip install --python $voicePython 'torch>=2.9.0' torchvision 'torchaudio>=2.9.0' `
    --index-url 'https://download.pytorch.org/whl/cu128'
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Stable CUDA 12.8 wheels were unavailable; trying the nightly channel.'
    & uv pip install --python $voicePython --pre 'torch>=2.9.0' torchvision 'torchaudio>=2.9.0' `
        --index-url 'https://download.pytorch.org/whl/nightly/cu128'
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install CUDA 12.8 PyTorch.' }
}

Write-Host 'Installing ASR/TTS dependencies...'
& uv pip install --python $voicePython -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Failed to install ASR/TTS dependencies.' }

Write-Host 'Installing Open-LLM-VTuber backend dependencies...'
Push-Location (Join-Path $script:PortableRoot 'app')
try {
    & uv sync --frozen
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
}
finally {
    Pop-Location
}

& "$PSScriptRoot\download-models.ps1" -VoicePython $voicePython
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }

Write-Host ''
Write-Host 'Installation completed. Run scripts\start-all.ps1.'
