param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '..\package\kajovochat_windows_apo_dev'),
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

$candidatePaths = @(
    $PackageDir,
    (Join-Path $PSScriptRoot $PackageDir)
)

$resolved = $null
foreach ($candidate in $candidatePaths) {
    if (Test-Path $candidate) {
        $resolved = (Resolve-Path $candidate).Path
        break
    }
}

if (-not $resolved) {
    throw "Balicek nebyl nalezen: $PackageDir"
}

$infPath = Join-Path $resolved 'kajovochat_windows_apo.inf'
if (!(Test-Path $infPath)) {
    throw "Chybi INF soubor: $infPath"
}

$command = "pnputil /add-driver `"$infPath`" /install"

if ($WhatIf) {
    Write-Host "WhatIf rezim:"
    Write-Host $command
    exit 0
}

Write-Host "Spoustim instalaci balicku:"
Write-Host $command
Start-Process -FilePath "pnputil.exe" -ArgumentList "/add-driver", $infPath, "/install" -Wait -NoNewWindow
