param(
    [string]$ProviderName = 'KajovoChat',
    [string]$OriginalFileName = 'kajovochat_windows_apo.inf',
    [ValidateSet('object', 'published-name')]
    [string]$OutputMode = 'object'
)

$ErrorActionPreference = 'Stop'

try {
    $drivers = Get-WindowsDriver -Online -All
} catch {
    throw "Dohledani balicku vyzaduje zvysena opravneni PowerShellu."
}

$matches = $drivers | Where-Object {
    $_.ProviderName -eq $ProviderName -and $_.OriginalFileName -eq $OriginalFileName
}

if ($OutputMode -eq 'published-name') {
    $matches | ForEach-Object { $_.Driver }
    exit 0
}

$matches | Select-Object ProviderName, OriginalFileName, Driver, ClassName, Date, Version
