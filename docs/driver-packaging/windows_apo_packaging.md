# Windows APO packaging

Datum: 2026-03-23

Tento dokument popisuje aktualni stav WDK/COM APO vetve pro KajovoChat.
Nejde o hotovy signed driver package, ale o pripraveny zaklad pro dalsi
nasazeni na Windows notebooku.

## Co je pripravene

- native helper DLL v `native/windows_aec_helper/`
- APO/COM helper kostra v `native/windows_apo_helper/`
- sdileny COM kontrakt v `native/windows_apo_helper/include/kajovochat_windows_apo_com.h`
- instalacni INF kostry v `native/windows_apo_helper/installer/kajovochat_windows_apo.dev.inf` a `native/windows_apo_helper/installer/kajovochat_windows_apo.prod.inf`
- staging skript v `native/windows_apo_helper/installer/package_windows_apo.ps1`
- VS Code tasky pro configure/build/test

## Co je porad jen kostra

- zadny podpisovy katalog
- zadny hotovy WHQL/attestation balicek
- zadna skutecna systemova registrace APO do konkretniho endpointu
- zadne ovladacove INF pravidlo pro produkcni nasazeni

## Jak se ma dal postupovat

1. Doplnit presne CLSID a endpoint binding podle ciloveho zarizeni.
2. Pripravit podpis a test na konkretnim notebooku.
3. Po uspechu nastavit `windows_native_preferred` jako prvni volbu s fallbackem na `webrtc_preferred`.
4. Pouzit staging skript pro konzistentni balicek DLL + INF + manifest.

## Ocekavany prubeh na Windows

- helper DLL se zkompiluje a otestuje lokalne
- dev/prod INF kostra se prevede na plny WDK balicek
- APO se zaregistruje do capture pipeline konkretniho zarizeni
- runtime probe v Pythonu pak pozna, jestli je dostupna nativni cesta
- staging skript vytvori balicek se soubory a sha256 manifestem

## Poznamka

Aktualni INF soubor je template a slouzi jako vychozi bod, ne jako hotove
produkcnı reseni.
