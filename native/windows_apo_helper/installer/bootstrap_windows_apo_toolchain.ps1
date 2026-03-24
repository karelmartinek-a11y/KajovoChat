param(
    [switch]$InstallMissing,
    [switch]$WhatIf = $true,
    [string]$WindowsSdkId = 'Microsoft.WindowsSDK.10.0.26100',
    [string]$WindowsWdkId = 'Microsoft.WindowsWDK.10.0.26100'
)

$ErrorActionPreference = 'Stop'

function Test-CommandAvailable {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

if (-not (Test-CommandAvailable -Name 'winget.exe')) {
    throw 'winget.exe nebyl nalezen. Bootstrap toolchain nelze provest.'
}

$auditPath = Join-Path $PSScriptRoot 'audit_windows_apo_prereqs.ps1'
$auditJson = & $auditPath -JsonOnly
$audit = $auditJson | ConvertFrom-Json

$commands = @()

if (-not $audit.checks.inf2cat.ok) {
    $commands += "winget install --id $WindowsSdkId --exact --accept-source-agreements --accept-package-agreements"
    $commands += "winget install --id $WindowsWdkId --exact --accept-source-agreements --accept-package-agreements"
}

if ($commands.Count -eq 0) {
    Write-Host 'Bootstrap toolchain: nic nechybi.'
    exit 0
}

Write-Host 'Bootstrap toolchain pripravil tyto prikazy:'
$commands | ForEach-Object { Write-Host " - $_" }

if (-not $InstallMissing -or $WhatIf) {
    Write-Host 'Rezim bez instalace. Pro skutecnou instalaci pouzij -InstallMissing -WhatIf:$false.'
    exit 0
}

foreach ($command in $commands) {
    Write-Host "Spoustim: $command"
    powershell -NoProfile -Command $command
}
