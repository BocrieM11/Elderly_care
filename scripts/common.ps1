$ErrorActionPreference = 'Stop'
$script:PortableRoot = Split-Path -Parent $PSScriptRoot

$localSettings = Join-Path $PSScriptRoot 'local-settings.ps1'
if (Test-Path -LiteralPath $localSettings) {
    . $localSettings
}

function Test-LocalPort {
    param([Parameter(Mandatory)][int]$Port)
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Resolve-VoicePython {
    $candidates = @(
        $env:OLV_VOICE_PYTHON,
        (Join-Path $script:PortableRoot 'runtime\voice-env\Scripts\python.exe')
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw 'Voice Python was not found. Run scripts\install.ps1 first.'
}

function Resolve-ModelPath {
    param(
        [string]$Override,
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string]$DisplayName
    )
    $path = if ($Override) { $Override } else { Join-Path $script:PortableRoot $RelativePath }
    if (-not (Test-Path -LiteralPath $path)) {
        throw "$DisplayName was not found at: $path. Run scripts\download-models.ps1 first."
    }
    return (Resolve-Path -LiteralPath $path).Path
}

function Wait-LocalPort {
    param(
        [Parameter(Mandatory)][int]$Port,
        [int]$TimeoutSeconds = 180
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-LocalPort -Port $Port) { return }
        Start-Sleep -Milliseconds 500
    }
    throw "Port $Port did not become ready within $TimeoutSeconds seconds."
}

function Test-HttpEndpoint {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [int]$TimeoutSeconds = 5
    )
    try {
        $null = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec $TimeoutSeconds
        return $true
    }
    catch {
        return $false
    }
}

function Wait-HttpEndpoint {
    param(
        [Parameter(Mandatory)][string]$Uri,
        [int]$TimeoutSeconds = 180
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-HttpEndpoint -Uri $Uri) { return }
        Start-Sleep -Milliseconds 1000
    }
    throw "Endpoint $Uri did not become ready within $TimeoutSeconds seconds."
}

function Test-RemoteLlm {
    if (-not $env:OLV_LLM_BASE_URL -or -not $env:OLV_LLM_MODEL) {
        throw 'OLV_LLM_BASE_URL and OLV_LLM_MODEL must be configured.'
    }

    $uri = $env:OLV_LLM_BASE_URL.TrimEnd('/') + '/models'
    try {
        $response = Invoke-RestMethod -Method Get -Uri $uri -TimeoutSec 10
    }
    catch {
        throw "Remote LLM is unavailable at ${uri}: $($_.Exception.Message)"
    }

    $modelIds = @($response.data | ForEach-Object { $_.id })
    if ($env:OLV_LLM_MODEL -notin $modelIds) {
        throw "Remote LLM model '$($env:OLV_LLM_MODEL)' was not found. Available models: $($modelIds -join ', ')"
    }
    return $true
}
