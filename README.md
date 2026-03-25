# Chatbot Kája

Desktop aplikace v `PySide6` pro hlasovou konverzaci přes OpenAI Realtime API.

## Co aplikace umí

- hands-free hlasovou konverzaci přes tlačítko `Start` v záhlaví
- bezpečné zadání a lokální uložení OpenAI API klíče přímo v hlavním okně
- EKG vizualizaci reagující na hlas a terminálový přepis posledních 10 řádků diskuze
- barge-in během mluvení asistenta
- adaptivní vícecestné tlumení self-hearing s odhadem latency, residual gate a double-talk ochranou
- malý audio selftest v záhlaví

## Požadavky

- Python `3.11+`
- funkční mikrofon a reproduktory nebo sluchátka
- platný OpenAI API klíč
## Instalace

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Spuštění

Hlavní aplikace:

```bash
python -m kajovochat
```

Alternativní vstupní bod:

```bash
python app_gui.py
```

Living orb demo:

```bash
python -m kajovochat.orb_demo
```

## OpenAI klíč

Po spuštění vložte OpenAI API klíč přímo do pole v záhlaví a potvrďte `Uložit klíč`. Klíč se ve Windows ukládá přes DPAPI, na ostatních platformách přes systémový keyring, pokud je dostupný.

## Konfigurace

Konfigurace se ukládá do `settings.json` v uživatelském profilu aplikace. Součástí konfigurace je i uložený audio guard profil a metadata poslední kalibrace podle páru zařízení. Technická dokumentace living orbu je v [docs/living_orb_renderer.md](docs/living_orb_renderer.md).

## Struktura projektu

- `kajovochat/` hlavní aplikační balíček
- `kajovochat/main.py` GUI a orchestrace hlasové relace
- `kajovochat/settings.py` minimální produktové nastavení a systémový prompt
- `kajovochat/services/` audio, realtime websocket, OpenAI služby a logování
- `kajovochat/widgets/` vlastní UI widgety
- `kajovochat/orb/` původní orb engine ponechaný kvůli kompatibilitě a testům starších částí
- `kajovochat/resources/assets/` obrázkové assety
- `tests/` testy

## Test a kontrola

```bash
python -m compileall -q kajovochat app_gui.py
pytest -q
```

## Audio AEC diagnostika

Aktualni audio stack pouziva kombinaci:

- vlastniho guardu a telemetrie v `kajovochat/main.py`
- nativni windows helper cesty `windows_system_aec`, pokud je dostupna DLL
- pripravene APO helper cesty `windows_system_aec`, pokud je dostupna DLL
- vlastniho adaptivniho AEC v `kajovochat/services/audio_service.py`
- volitelneho backendu `aec-audio-processing` pro WebRTC AEC cisteni vhodnych echo-only bloku

Pri realnem ladeni na HW je dulezite sledovat session `jsonl` log v adresari logu aplikace a hlavne zaznamy `aec_diag` a `aec_summary`.

Nejdolezitejsi pole:

- `reference_miss_ratio`: jak casto AEC vubec nemel pouzitelnou playback reference
- `reference_ready_ratio`: jak casto byla reference pripravena
- `aligned_ratio`: jak casto se reference a mic chunk rozumne zarovnaly
- `avg_quality_when_aligned`: kvalita odeectu jen v zarovnanych blocich
- `avg_delay_error`: prumerna chyba mezi runtime delay a aktualni kalibraci
- `backend`: jestli blok cistil `custom` nebo `webrtc`
- `ws=on`: `webrtc_success`, tedy blok, kde WebRTC backend realne zafungoval i kdyz vlastni `similarity` nemusela byt vysoka

Prakticka interpretace:

- vysoke `reference_miss_ratio` znamena problem v playback reference pipeline
- nizke `aligned_ratio` znamena problem v coarse delay/alignment vrstve
- `backend=webrtc` + `ws=on` + nizky `residual` znamena realne funkcni echo odeect
- vysoke `avg_delay_error` znamena nestabilni runtime latenci

Aktualni stav projektu je prakticky pouzitelny, ale stale nejde o plne stabilni OS-level AEC. Dalsi ladeni ma smysl delat podle realnych session logu, ne naslepo.

## Session log troubleshooting

Session `.jsonl` log je teď primární zdroj pravdy pro notebookovou hlasovou relaci. Při ladění jedné relace sledujte minimálně:

- `audio_session_state` – lifecycle relace (`starting` → `probing` → `active` / `degraded` / `recovering`)
- `audio_backend_selected` – požadovaný a skutečně aktivní backend
- `audio_backend_fallback` – řízený přechod na další backend v chainu
- `audio_reference_health` – stav playback reference pipeline
- `reconnect_*` – transport recovery
- `aec_diag` a `aec_summary` – block-level DSP diagnostika

Bezpečný default konfigurace je:

- `audio_aec_mode=windows_system_aec`
- `audio_device_mode=auto`
- `audio_session_profile=production`
- `audio_diagnostics_enabled=false`

Diagnostické přepínače jsou oddělené od produkčních režimů. `custom_lab` je explicitní laboratorní mód, ne běžná produkční cesta.

## Build pro macOS

Pro vytvoření `.app` balíčku je v repu připravený PyInstaller build skript:

```bash
python3 tools/build_macos_app.py
```

Výstup najdeš v `dist/ChatbotKaja.app`.

Build používá `app_gui.py` jako vstupní bod, protože je to bezpečnější top-level script pro PyInstaller než balení `kajovochat/__main__.py`.
