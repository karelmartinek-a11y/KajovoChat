param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '..\package\kajovochat_windows_apo_production'),
    [string]$CertificateThumbprint,
    [string]$TimestampUrl = 'http://timestamp.digicert.com',
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

function Resolve-PackageDir {
    param([string]$Value)

    $candidatePaths = @(
        $Value,
        (Join-Path $PSScriptRoot $Value)
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Balicek nebyl nalezen: $Value"
}

function Find-SignTool {
    $candidates = @(
        (Get-Command signtool.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        'C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe',
        'C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe',
        'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe'
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw 'signtool.exe nebyl nalezen. Nainstaluj Windows SDK nebo dopln PATH.'
}

$resolved = Resolve-PackageDir -Value $PackageDir
$dllPath = Join-Path $resolved 'kajovochat_windows_apo.dll'
$infPath = Join-Path $resolved 'kajovochat_windows_apo.inf'
$catPath = Join-Path $resolved 'kajovochat_windows_apo.cat'

if (!(Test-Path $dllPath)) {
    throw "Chybi DLL: $dllPath"
}

if (!(Test-Path $infPath)) {
    throw "Chybi INF: $infPath"
}

$signtool = Find-SignTool

if ([string]::IsNullOrWhiteSpace($CertificateThumbprint)) {
    throw 'Chybi CertificateThumbprint pro podpis balicku.'
}

$targets = @($dllPath)
if (Test-Path $catPath) {
    $targets += $catPath
}

if ($WhatIf) {
    Write-Host 'WhatIf rezim:'
    foreach ($target in $targets) {
        Write-Host "$signtool sign /sha1 $CertificateThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $target"
    }
    exit 0
}

foreach ($target in $targets) {
    & $signtool sign /sha1 $CertificateThumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $target
}
