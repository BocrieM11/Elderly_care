. "$PSScriptRoot\common.ps1"
$env:PYTHONUTF8 = '1'

if (Test-LocalPort -Port 50000) {
    if (Test-HttpEndpoint -Uri 'http://127.0.0.1:50000/v1/health') {
        Write-Host 'Fish Speech TTS is already running on port 50000.'
        exit 0
    }
    throw 'Port 50000 is occupied by a service that is not Fish Speech.'
}

$python = $env:OLV_FISH_PYTHON
if (-not $python -or -not (Test-Path -LiteralPath $python)) {
    throw "Fish Speech Python was not found: $python"
}
$sourceRoot = $env:OLV_FISH_SOURCE
if (-not $sourceRoot -or -not (Test-Path -LiteralPath $sourceRoot)) {
    throw "Fish Speech source was not found: $sourceRoot"
}
$serviceRoot = Join-Path $script:PortableRoot 'services\fishspeech'
$referenceSource = Join-Path $serviceRoot 'references'
$referenceTarget = Join-Path $sourceRoot 'references'
if (Test-Path -LiteralPath $referenceSource) {
    New-Item -ItemType Directory -Path $referenceTarget -Force | Out-Null
    Get-ChildItem -LiteralPath $referenceSource -Directory | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $referenceTarget -Recurse -Force
    }
}
$model = Resolve-ModelPath -Override $env:OLV_FISH_MODEL `
    -RelativePath 'models\fish-speech-1.5' `
    -DisplayName 'Fish Speech 1.5 model'
$decoder = Join-Path $model 'firefly-gan-vq-fsq-8x1024-21hz-generator.pth'
if (-not (Test-Path -LiteralPath $decoder)) {
    throw "Fish Speech decoder was not found: $decoder"
}
$env:PYTHONPATH = "$serviceRoot;$sourceRoot"

Write-Host "Starting Fish Speech 1.5 with model: $model"
Push-Location $sourceRoot
try {
    & $python -m tools.api_server `
        --listen 127.0.0.1:50000 `
        --llama-checkpoint-path $model `
        --decoder-checkpoint-path $decoder `
        --decoder-config-name firefly_gan_vq `
        --device cuda `
        --half `
        --compile `
        --workers 1
    if ($LASTEXITCODE -ne 0) { throw "Fish Speech exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
