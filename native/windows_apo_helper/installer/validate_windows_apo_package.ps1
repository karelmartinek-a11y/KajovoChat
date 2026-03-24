param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '..\package\kajovochat_windows_apo_dev')
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

$required = @(
    'kajovochat_windows_apo.dll',
    'kajovochat_windows_apo.inf',
    'README.md',
    'package-manifest.json'
)

$missing = @()
foreach ($item in $required) {
    $path = Join-Path $resolved $item
    if (!(Test-Path $path)) {
        $missing += $item
    }
}

if ($missing.Count -gt 0) {
    Write-Host "Chybi soubory:"
    $missing | ForEach-Object { Write-Host " - $_" }
    exit 1
}

$manifestPath = Join-Path $resolved 'package-manifest.json'
$manifest = Get-Content -Path $manifestPath -Raw | ConvertFrom-Json

Write-Host "Balicek je konzistentni."
if ($manifest.profile) {
    Write-Host "Profil: $($manifest.profile)"
}
$manifestFiles = @{}
foreach ($entry in $manifest.files) {
    $manifestFiles[$entry.name] = $entry.sha256
}

if ($manifestFiles.ContainsKey('kajovochat_windows_apo.dll')) {
    Write-Host "DLL SHA256: $($manifestFiles['kajovochat_windows_apo.dll'])"
}

if ($manifestFiles.ContainsKey('kajovochat_windows_apo.inf')) {
    Write-Host "INF SHA256: $($manifestFiles['kajovochat_windows_apo.inf'])"
}

if ($manifestFiles.ContainsKey('kajovochat_windows_apo.cat')) {
    Write-Host "CAT SHA256: $($manifestFiles['kajovochat_windows_apo.cat'])"
}

Write-Host "Dalsi krok: podpis, WDK packaging a test na cilovem notebooku."
