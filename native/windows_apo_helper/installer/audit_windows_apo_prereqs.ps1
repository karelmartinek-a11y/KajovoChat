param(
    [string]$PackageDir = (Join-Path $PSScriptRoot '..\package\kajovochat_windows_apo_production'),
    [switch]$JsonOnly
)

$ErrorActionPreference = 'Stop'

function Test-CandidatePath {
    param([string[]]$Candidates)

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

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

    return $null
}

$resolvedPackage = Resolve-PackageDir -Value $PackageDir
$signtoolPath = Test-CandidatePath -Candidates @(
    (Get-Command signtool.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    'C:\Program Files (x86)\Windows Kits\10\App Certification Kit\signtool.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\x64\signtool.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe'
)

$inf2catPath = Test-CandidatePath -Candidates @(
    (Get-Command Inf2Cat.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x86\Inf2Cat.exe',
    'C:\Program Files (x86)\Windows Kits\10\Tools\x64\Inf2Cat.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\x64\Inf2Cat.exe',
    'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\Inf2Cat.exe'
)

$cmakePath = Test-CandidatePath -Candidates @(
    (Get-Command cmake.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
)

$pnputilPath = Test-CandidatePath -Candidates @(
    (Get-Command pnputil.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    'C:\Windows\System32\pnputil.exe'
)

$buildToolsPath = Test-CandidatePath -Candidates @(
    'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools',
    'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
)

$packageCat = if ($resolvedPackage) { Test-Path (Join-Path $resolvedPackage 'kajovochat_windows_apo.cat') } else { $false }
$packageInf = if ($resolvedPackage) { Test-Path (Join-Path $resolvedPackage 'kajovochat_windows_apo.inf') } else { $false }
$packageDll = if ($resolvedPackage) { Test-Path (Join-Path $resolvedPackage 'kajovochat_windows_apo.dll') } else { $false }

$report = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    package_dir = $resolvedPackage
    checks = [ordered]@{
        build_tools = [ordered]@{ ok = [bool]$buildToolsPath; path = $buildToolsPath }
        cmake = [ordered]@{ ok = [bool]$cmakePath; path = $cmakePath }
        signtool = [ordered]@{ ok = [bool]$signtoolPath; path = $signtoolPath }
        inf2cat = [ordered]@{ ok = [bool]$inf2catPath; path = $inf2catPath }
        pnputil = [ordered]@{ ok = [bool]$pnputilPath; path = $pnputilPath }
        package_inf = [ordered]@{ ok = [bool]$packageInf }
        package_dll = [ordered]@{ ok = [bool]$packageDll }
        package_cat = [ordered]@{ ok = [bool]$packageCat }
    }
}

$missing = @()
foreach ($entry in $report.checks.GetEnumerator()) {
    if (-not $entry.Value.ok) {
        $missing += $entry.Key
    }
}

if (-not $JsonOnly) {
    if ($missing.Count -eq 0) {
        Write-Host 'APO prerequisite audit: vse dostupne.'
    } else {
        Write-Host 'APO prerequisite audit: chybi nebo neni pripraveno:'
        $missing | ForEach-Object { Write-Host " - $_" }
    }
}

$report | ConvertTo-Json -Depth 5 -Compress:$JsonOnly
