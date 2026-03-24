# Windows APO deployment

Datum: 2026-03-24

Tento dokument shrnuje, co je v repozitari pripravene pro WDK/COM APO cestu a
co jeste chybi pro skutecne nasazeni na notebooku s Windows.

## Co je uz hotove

- native helper DLL v `native/windows_aec_helper/`
- APO/COM helper skeleton v `native/windows_apo_helper/`
- COM kontrakt a exporty v `native/windows_apo_helper/include/kajovochat_windows_apo_com.h`
- staging balicek skript v `native/windows_apo_helper/installer/package_windows_apo.ps1`
- INF templaty pro APO registraci v `native/windows_apo_helper/installer/kajovochat_windows_apo.dev.inf` a `native/windows_apo_helper/installer/kajovochat_windows_apo.prod.inf`
- instalacni a odinstalacni skripty v `native/windows_apo_helper/installer/install_windows_apo_package.ps1` a `native/windows_apo_helper/installer/uninstall_windows_apo_package.ps1`
- dohledani nainstalovaneho balicku v `native/windows_apo_helper/installer/find_windows_apo_package.ps1`
- release preflight a signing skeleton v `native/windows_apo_helper/installer/preflight_windows_apo_release.ps1` a `native/windows_apo_helper/installer/sign_windows_apo_package.ps1`
- generovani katalogu skeletonem v `native/windows_apo_helper/installer/new_windows_apo_catalog.ps1`
- audit prerequisite nastroju v `native/windows_apo_helper/installer/audit_windows_apo_prereqs.ps1`
- bootstrap chybejicich SDK/WDK nastroju v `native/windows_apo_helper/installer/bootstrap_windows_apo_toolchain.ps1`

## Jak je resena registrace

INF templaty uz pouzivaji oficialni Windows APO property keys pro mode effect:

- `PKEY_FX_ModeEffectClsid`
- `PKEY_MFX_ProcessingModes_Supported_For_Streaming`

V production template jsou pripravene i komentovane EFX varianty pro pripad, ze bude
potreba zvlast registrovat APO jako endpoint effect.

## Co je jeste nutne doplnit pred nasazenim

1. Vybrat konkretni cilovy model notebooku nebo driver stack.
2. Doplnit podpisovy katalog a WDK build.
3. Overit, jestli ma byt APO zaregistrovane jako MFX nebo EFX.
4. Doplnit skutecny binding na endpoint / capture pipeline.
5. Otestovat instalaci na ciste stanici s logovanim registry a audio probe.

## Minimalni deploy flow

1. Spustit `Native APO: Build`.
2. Spustit `Native APO: Package` nebo `Native APO: Package Production`.
3. Zkontrolovat manifest v `native/windows_apo_helper/package/`.
4. Spustit audit prerequisite nastroju.
5. Spustit odpovidajici validacni task.
6. Spustit release preflight.
7. Vygenerovat `.cat` pres Inf2Cat.
8. Podepsat DLL/INF podle ciloveho deployment modu.
9. Otestovat instalacni task v `WhatIf` rezimu.
10. Provest instalaci na testovacim zarizeni.

## Instalace a odinstalace

- instalace je pripravena pres `pnputil /add-driver ... /install`
- odinstalace je pripravena pres `pnputil /delete-driver ... /uninstall /force`
- publikovany nazev `oemXX.inf` lze dohledat skriptem nad `Get-WindowsDriver -Online -All`
- skripty umi `-WhatIf`, aby slo nejdriv overit presny prikaz bez zasahu do systemu
- release preflight umi zkontrolovat balicek, `.cat` a `signtool.exe`
- release preflight umi zkontrolovat i `Inf2Cat.exe`
- katalog lze pripravit pres `Inf2Cat` skeleton skript
- signing skeleton umi pripravit prikazy pro podpis DLL i `.cat` podle thumbprintu certifikatu
- prerequisite audit umi v jednom vystupu zkontrolovat Build Tools, CMake, signtool, Inf2Cat, pnputil i stav balicku
- package manifest nese SHA256 i pro `.cat`, pokud uz byl katalog vygenerovan
- bootstrap skript umi pripravit `winget install` prikazy pro Windows SDK a WDK, kdyz chybi `Inf2Cat`

## Poznamka k opravnenim

Skript pro dohledani nainstalovaneho balicku pouziva `Get-WindowsDriver -Online -All`
a vyzaduje zvysena opravneni PowerShellu.

## Dulezita poznamka

Tohle stale neni hotovy produkcni APO balicek. Je to ale uz konfiguracne
spravne zarovnany zaklad, ktery odpovida tomu, jak Windows APO registry a
property store ocekavaji zapis effect CLSID.
