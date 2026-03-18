# Forenzní hluboký audit KajovoChat

Datum auditu: 2026-03-18
Repozitář: `C:\GitHub\KajovoChat`
Stav: po zapracování auditních nálezů do kódu a dokumentace

## Shrnutí

Původní nálezy z předchozí verze auditu byly zapracovány do runtime, nastavení, logování, testů i dokumentace. Aplikace nyní funguje jako obecný hlasový asistent nad OpenAI Realtime API bez individuální autorizace, s jednotným bootstrapem, bezpečnějším lokálním uložením API klíče, vypnutým logováním obsahu konverzace v defaultu a s reconnect logikou pro realtime relaci.

Aktuální otevřený bod už není kódový, ale repozitářový: ve `git status` jsou stále připravené mazací změny historických cizích souborů a starých logů, které je potřeba commitnout, aby stav pracovního stromu odpovídal i poslednímu commitu.

## Ověření

- `pytest -q`: PASS, `4 passed`
- `python -m compileall -q kajovochat app_gui.py`: PASS
- headless Qt smoke: PASS, hlavní okno vytvořeno
- textové soubory: UTF-8 bez BOM

## Zapracované opravy

### Stabilita a provoz

- doplněn reconnect realtime websocketu s backoffem a limitem pokusů
- doplněna základní telemetry relace:
  - `speech_started`
  - `speech_stopped`
  - `assistant_audio_first_delta`
  - `response_done`
- bootstrap `app_gui.py` sjednocen s `python -m kajovochat`
- `showMaximized()` přesunuto z konstruktoru okna do vstupního bodu

### Bezpečnost a privacy

- OpenAI API klíč je ve Windows ukládán přes DPAPI
- fallback na starou obfuskaci zůstává jen pro kompatibilitu nebo nepodporované platformy
- do nastavení přidán přepínač `Ukládat obsah konverzace do logů`
- výchozí stav je privacy-safe: obsah relace se neloguje, ukládají se jen technická metadata

### Konzistence a UX

- sjednocen limit TTS rychlosti mezi GUI a runtime na `0.25–1.5`
- dialog nastavení už neprezentuje nepoužívaný chat model jako hlavní runtime volbu
- v captions se nově zobrazuje aktivní model, hlas a režim logování
- README doplněn o fyzické umístění `settings.json` a chování ukládání klíče

### Robustnost logování

- `RealtimeLogWriter` má omezenou frontu, eviduje dropped records a poslední chybu
- logger při zavírání dopumpuje zbytek fronty, aby se neztrácel konec relace
- worker umí upozornit do captions, když logování narazí na chybu

### Testovatelnost

- placeholder test odstraněn
- doplněny reálné testy pro:
  - settings migraci a prompt
  - parser eventů realtime služby
  - log writer a korektní zápis souborů

## Potenciál

Níže jsou už jen rozvojové body, ne neopravené vady:

- přidat mock websocket server pro integrační testy bez sítě
- rozšířit stavový model UI o explicitní `connecting` a `reconnecting`
- přidat redakci citlivých textů i do obecných runtime chyb
- doplnit první-run onboarding pro API klíč, mikrofon a reproduktor
- commitnout připravené mazací změny, aby audit platil i pro poslední commit v historii

## Závěr

Kódové a dokumentační nálezy z předchozího auditu jsou zapracované. Aktuální stav je technicky konzistentní, ověřený testy a připravený na další vývoj. Jediný zbývající otevřený bod je commitnutí už připraveného repozitářového úklidu.
