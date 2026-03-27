$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Resolve-PythonCommand {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        try {
            & py -3.11 -V *> $null
            return @('py', '-3.11')
        } catch {
            return @('py')
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @('python')
    }

    throw 'Python 3.11+ nebyl nalezen. Nainstaluj Python a spust skript znovu.'
}

$pythonCmd = Resolve-PythonCommand
$venvDir = Join-Path $root '.venv'
$venvPython = Join-Path $venvDir 'Scripts\python.exe'
$requirements = Join-Path $root 'requirements.txt'
$marker = Join-Path $venvDir '.kajovochat_requirements_installed'

if (-not (Test-Path $venvPython)) {
    Write-Host '[INFO] Vytvarim virtualni prostredi...'
    & $pythonCmd[0] $pythonCmd[1..($pythonCmd.Length-1)] -m venv $venvDir
}

if (-not (Test-Path $venvPython)) {
    throw 'Virtualni prostredi se nepodarilo vytvorit.'
}

$installDeps = $false
if (-not (Test-Path $marker)) {
    $installDeps = $true
} elseif ((Get-Item $requirements).LastWriteTimeUtc -gt (Get-Item $marker).LastWriteTimeUtc) {
    $installDeps = $true
}

if ($installDeps) {
    Write-Host '[INFO] Instaluji nebo aktualizuji zavislosti...'
    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r $requirements
    Set-Content -Path $marker -Value 'installed' -Encoding utf8
}

Write-Host '[INFO] Spoustim Chatbot Kaja...'
& $venvPython -m kajovochat
