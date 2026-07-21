param(
    [string]$VoicePython
)

. "$PSScriptRoot\common.ps1"

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Ollama is not installed or not available in PATH.'
}

if (-not $VoicePython) {
    $VoicePython = Resolve-VoicePython
}

$modelRoot = Join-Path $script:PortableRoot 'models'
$cosyVoiceModel = Join-Path $modelRoot 'CosyVoice2-0.5B'
$funAsrModel = Join-Path $modelRoot 'Fun-ASR-Nano-2512'
$funAsrVadModel = Join-Path $modelRoot 'FSMN-VAD'
$kwsModelName = 'sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20'
$kwsModel = Join-Path $modelRoot $kwsModelName
New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null

Write-Host 'Pulling qwen3.5:4b with Ollama...'
& ollama pull qwen3.5:4b
if ($LASTEXITCODE -ne 0) { throw 'Failed to pull qwen3.5:4b.' }

$downloadCode = @'
import argparse
from modelscope import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--cosyvoice", required=True)
parser.add_argument("--funasr", required=True)
parser.add_argument("--vad", required=True)
args = parser.parse_args()

snapshot_download("iic/CosyVoice2-0.5B", local_dir=args.cosyvoice)
snapshot_download("FunAudioLLM/Fun-ASR-Nano-2512", local_dir=args.funasr)
snapshot_download("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", local_dir=args.vad)
'@

Write-Host 'Downloading CosyVoice2 and Fun-ASR models from ModelScope...'
& $VoicePython -c $downloadCode --cosyvoice $cosyVoiceModel --funasr $funAsrModel --vad $funAsrVadModel
if ($LASTEXITCODE -ne 0) { throw 'Failed to download ModelScope models.' }

if (-not (Test-Path -LiteralPath (Join-Path $kwsModel 'tokens.txt'))) {
    $kwsArchive = Join-Path $modelRoot "$kwsModelName.tar.bz2"
    $kwsUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$kwsModelName.tar.bz2"
    Write-Host 'Downloading the sherpa-onnx streaming keyword-spotting model...'
    Invoke-WebRequest -Uri $kwsUrl -OutFile $kwsArchive
    & tar.exe -xjf $kwsArchive -C $modelRoot
    if ($LASTEXITCODE -ne 0) { throw 'Failed to extract the KWS model.' }
    Remove-Item -LiteralPath $kwsArchive -Force
}

$kwsCli = Join-Path (Split-Path -Parent $VoicePython) 'sherpa-onnx-cli.exe'
$keywordsRaw = Join-Path $script:PortableRoot 'services\funasr\keywords_raw.txt'
$keywordsFile = Join-Path $script:PortableRoot 'services\funasr\keywords.txt'
if (-not (Test-Path -LiteralPath $kwsCli)) {
    throw "sherpa-onnx-cli was not found: $kwsCli"
}
Write-Host 'Generating KWS tokens for the three Chinese wake phrases...'
& $kwsCli text2token --tokens (Join-Path $kwsModel 'tokens.txt') `
    --tokens-type 'phone+ppinyin' --lexicon (Join-Path $kwsModel 'en.phone') `
    $keywordsRaw $keywordsFile
if ($LASTEXITCODE -ne 0) { throw 'Failed to generate the KWS keywords file.' }

Write-Host 'All models are ready.'
