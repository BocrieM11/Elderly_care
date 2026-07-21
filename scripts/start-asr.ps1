. "$PSScriptRoot\common.ps1"
$env:PYTHONUTF8 = '1'
$env:CUDA_VISIBLE_DEVICES = '0'

if (Test-LocalPort -Port 10095) {
    Write-Host 'Fun-ASR is already running on port 10095.'
    exit 0
}

$python = Resolve-VoicePython
$serviceRoot = Join-Path $script:PortableRoot 'services\funasr'
$server = Join-Path $serviceRoot 'serve_olv.py'
$model = Resolve-ModelPath -Override $env:OLV_FUNASR_MODEL `
    -RelativePath 'models\Fun-ASR-Nano-2512' `
    -DisplayName 'Fun-ASR model'

$device = if ($env:OLV_ASR_DEVICE) { $env:OLV_ASR_DEVICE } else { 'cuda:0' }
$vadModel = if ($env:OLV_FUNASR_VAD_MODEL) {
    $env:OLV_FUNASR_VAD_MODEL
}
else {
    $portableVad = Join-Path $script:PortableRoot 'models\FSMN-VAD'
    if (Test-Path -LiteralPath $portableVad) { $portableVad } else { 'fsmn-vad' }
}
$vadDevice = if ($env:OLV_VAD_DEVICE) { $env:OLV_VAD_DEVICE } else { 'cpu' }
$kwsModel = Resolve-ModelPath -Override $env:OLV_KWS_MODEL `
    -RelativePath 'models\sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20' `
    -DisplayName 'sherpa-onnx KWS model'
$kwsKeywords = Join-Path $serviceRoot 'keywords.txt'
if (-not (Test-Path -LiteralPath $kwsKeywords)) {
    throw "KWS keywords file was not found: $kwsKeywords"
}
Write-Host "Starting Fun-ASR on $device with model: $model"
Write-Host "Starting FSMN-VAD on $vadDevice with model: $vadModel"
Write-Host "Starting sherpa-onnx KWS on CPU with model: $kwsModel"
Push-Location $serviceRoot
try {
    & $python $server --model $model --host 127.0.0.1 --port 10095 --device $device `
        --vad-model $vadModel --vad-device $vadDevice `
        --kws-model-dir $kwsModel --kws-keywords-file $kwsKeywords
    if ($LASTEXITCODE -ne 0) { throw "Fun-ASR exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
