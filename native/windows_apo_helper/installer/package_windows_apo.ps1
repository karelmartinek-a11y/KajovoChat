param(
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\build\bin\Release'),
    [string]$OutputDir = (Join-Path $PSScriptRoot '..\package'),
    [ValidateSet('dev', 'production')]
    [string]$Profile = 'dev',
    [switch]$RefreshOnly
)

$ErrorActionPreference = 'Stop'

function Write-Utf8NoBomText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function New-ManifestFileEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    return [ordered]@{
        name = $Name
        sha256 = (Get-FileHash -Path $Path -Algorithm SHA256).Hash
    }
}

$buildRoot = (Resolve-Path $BuildDir).Path
$outputRoot = [System.IO.Path]::GetFullPath($OutputDir)
$bundleDir = Join-Path $outputRoot ("kajovochat_windows_apo_" + $Profile)
$manifestPath = Join-Path $bundleDir 'package-manifest.json'

$dllPath = Join-Path $buildRoot 'kajovochat_windows_apo.dll'
$infFileName = if ($Profile -eq 'production') { 'kajovochat_windows_apo.prod.inf' } else { 'kajovochat_windows_apo.dev.inf' }
$infPath = Join-Path $PSScriptRoot $infFileName
$readmePath = Join-Path $PSScriptRoot '..\README.md'

if (!(Test-Path $dllPath) -and -not $RefreshOnly) {
    throw "Chybi build DLL: $dllPath"
}

New-Item -ItemType Directory -Force -Path $bundleDir | Out-Null

if (-not $RefreshOnly) {
    Copy-Item -Force $dllPath (Join-Path $bundleDir 'kajovochat_windows_apo.dll')
    Copy-Item -Force $infPath (Join-Path $bundleDir 'kajovochat_windows_apo.inf')
    Copy-Item -Force $readmePath (Join-Path $bundleDir 'README.md')
}

$files = @(
    (New-ManifestFileEntry -Path (Join-Path $bundleDir 'kajovochat_windows_apo.dll') -Name 'kajovochat_windows_apo.dll'),
    (New-ManifestFileEntry -Path (Join-Path $bundleDir 'kajovochat_windows_apo.inf') -Name 'kajovochat_windows_apo.inf')
)

$catPath = Join-Path $bundleDir 'kajovochat_windows_apo.cat'
if (Test-Path $catPath) {
    $files += (New-ManifestFileEntry -Path $catPath -Name 'kajovochat_windows_apo.cat')
}

$manifest = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    profile = $Profile
    build_dir = $buildRoot
    bundle_dir = $bundleDir
    files = $files
    note = 'Tento balicek je pouze staging skeleton. Pred instalaci dopln podpis, WDK balicek a skutecnou registraci APO.'
}

$manifestJson = $manifest | ConvertTo-Json -Depth 4
Write-Utf8NoBomText -Path $manifestPath -Text $manifestJson

if ($RefreshOnly) {
    Write-Host "Manifest obnoven: $bundleDir"
} else {
    Write-Host "Staging hotov: $bundleDir"
}
Write-Host "Manifest: $manifestPath"
