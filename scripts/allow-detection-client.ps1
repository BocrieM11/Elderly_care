param(
    [Parameter(Mandatory)]
    [string]$DetectionComputerIp
)

$ErrorActionPreference = 'Stop'

try {
    $parsedIp = [System.Net.IPAddress]::Parse($DetectionComputerIp)
}
catch {
    throw "Invalid detection computer IP address: $DetectionComputerIp"
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run PowerShell as Administrator before executing this script.'
}

$ruleName = 'OLV Detection Events 12393'
$existingRule = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existingRule) {
    Remove-NetFirewallRule -DisplayName $ruleName
}

New-NetFirewallRule `
    -DisplayName $ruleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 12393 `
    -RemoteAddress $parsedIp.IPAddressToString `
    -Profile Any | Out-Null

Write-Host "Allowed $($parsedIp.IPAddressToString) to access OLV TCP port 12393."
