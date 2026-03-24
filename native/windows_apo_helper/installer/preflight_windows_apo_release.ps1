param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '..\package\kajovochat_windows_apo_production')
)

$ErrorActionPreference = 'Stop'

$validator = Join-Path $PSScriptRoot 'validate_windows_apo_package.ps1'
& $validator -PackageDir $PackageDir

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

$catPath = Join-Path $resolved 'kajovochat_windows_apo.cat'
if (!(Test-Path $catPath)) {
    Write-Host 'Varovani: chybi katalog .cat. Produkcni instalace bez nej nebude kompletni.'
}

$inf2catFound = $false
$inf2catCandidates = @(
    (Get-Command Inf2Cat.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86\Inf2Cat.exe',
    'C:\Program Files (x86)\Windows Kits\10\Tools\x64\Inf2Cat.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\x64\Inf2Cat.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\Inf2Cat.exe'
) | Where-Object { $_ }

foreach ($candidate in $inf2catCandidates) {
    if (Test-Path $candidate) {
        $inf2catFound = $true
        Write-Host "Inf2Cat nalezen: $candidate"
        break
    }
}

if (-not $inf2catFound) {
    Write-Host 'Varovani: Inf2Cat.exe nebyl nalezen.'
}

$signtoolFound = $false
$signtoolCandidates = @(
    (Get-Command signtool.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    'C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe'
) | Where-Object { $_ }

foreach ($candidate in $signtoolCandidates) {
    if (Test-Path $candidate) {
        $signtoolFound = $true
        Write-Host "signtool nalezen: $candidate"
        break
    }
}

if (-not $signtoolFound) {
    Write-Host 'Varovani: signtool.exe nebyl nalezen.'
}

Write-Host 'Preflight hotov.'
