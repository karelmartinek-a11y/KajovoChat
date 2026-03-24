param(
    [string]$PublishedName,
    [string]$ProviderName = 'KajovoChat',
    [string]$OriginalFileName = 'kajovochat_windows_apo.inf',
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($PublishedName)) {
    $finderPath = Join-Path $PSScriptRoot 'find_windows_apo_package.ps1'
    $detected = & $finderPath -ProviderName $ProviderName -OriginalFileName $OriginalFileName -OutputMode published-name
    if (!$detected) {
        throw "Nepodarilo se dohledat publikovany nazev driveru pro provider '$ProviderName'."
    }
    $PublishedName = ($detected | Select-Object -First 1).ToString().Trim()
}

$command = "pnputil /delete-driver `"$PublishedName`" /uninstall /force"

if ($WhatIf) {
    Write-Host "WhatIf rezim:"
    Write-Host $command
    exit 0
}

Write-Host "Spoustim odinstalaci balicku:"
Write-Host $command
Start-Process -FilePath "pnputil.exe" -ArgumentList "/delete-driver", $PublishedName, "/uninstall", "/force" -Wait -NoNewWindow
