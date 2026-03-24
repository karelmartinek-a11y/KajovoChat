param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '..\package\kajovochat_windows_apo_production'),
    [string]$OsList = '10_X64',
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

function Find-Inf2Cat {
    $candidates = @(
        (Get-Command Inf2Cat.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86\Inf2Cat.exe',
        'C:\Program Files (x86)\Windows Kits\10\Tools\x64\Inf2Cat.exe',
        'C:\Program Files (x86)\Windows Kits\10\bin\x64\Inf2Cat.exe',
        'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\Inf2Cat.exe'
    ) | Where-Object { $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw 'Inf2Cat.exe nebyl nalezen. Nainstaluj WDK/Windows SDK Tools.'
}

$resolved = Resolve-PackageDir -Value $PackageDir
$infPath = Join-Path $resolved 'kajovochat_windows_apo.inf'

if (!(Test-Path $infPath)) {
    throw "Chybi INF: $infPath"
}

$inf2cat = Find-Inf2Cat
$command = "`"$inf2cat`" /driver:`"$resolved`" /os:$OsList"

if ($WhatIf) {
    Write-Host 'WhatIf rezim:'
    Write-Host $command
    exit 0
}

Start-Process -FilePath $inf2cat -ArgumentList "/driver:$resolved", "/os:$OsList" -Wait -NoNewWindow
