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
- `kajovochat/main.py` GUI vrstva a delegace na session entry pointy
- `kajovochat/audio/` finální session-oriented audio architektura
- `kajovochat/settings.py` minimální produktové nastavení a systémový prompt
- `kajovochat/services/` podpůrné služby bez audio runtime ownershipu
- `kajovochat/widgets/` vlastní UI widgety
- `kajovochat/orb/` původní orb engine ponechaný kvůli kompatibilitě a testům starších částí
- `kajovochat/resources/assets/` obrázkové assety
- `tests/` testy

## Test a kontrola

```bash
python -m compileall -q kajovochat app_gui.py
pytest -q
```

## Finální audio architektura

Audio stack je po etapě 8 uzavřený jako session-oriented architektura v `kajovochat/audio/`.

Hlavní autority jsou:

- `AudioSessionManager` pro session entry pointy a aplikaci rozhodnutí
- `VoiceGate` pro hlasovou UX politiku
- `RecoverySupervisor` pro recovery policy
- `AudioTelemetry` pro session health, fallback story a serializovatelný snapshot
- `windows_system_aec` jako finálně uzavřený produkční backend detail

Produktové režimy jsou explicitní a auditovatelné:

- `notebook_builtin + windows_system_aec`
- `notebook_builtin + webrtc_apm`
- `notebook_builtin + degraded_no_aec`
- `headset_clean`

Deterministický fallback chain je:

- `windows_system_aec -> webrtc_apm -> degraded_no_aec`
- `headset_clean` je samostatná topology-locked cesta

Pro důkazy a reprodukovatelné běhy viz:

- `docs/final_audio_architecture.md`
- `FINAL_ACCEPTANCE_MATRIX.md`
- `docs/audio_acceptance_evidence/`
- `tools/audio_architecture_harness.py`
- `tools/generate_audio_acceptance_evidence.py`

## Session log troubleshooting

Session `.jsonl` log je primární zdroj pravdy pro notebookovou hlasovou relaci. Sledujte minimálně:

- `audio_session_state` – lifecycle relace (`idle` → `starting` → `probing` → `active` / `degraded` / `recovering` → `stopping` / `failed`)
- `audio_backend_selected` – požadovaný a skutečně aktivní backend
- `audio_backend_fallback` – řízený přechod na další backend v chainu
- `audio_reference_health` – stav playback reference pipeline
- `reconnect_*` – transport recovery
- `session_telemetry_snapshot` – serializovatelný session snapshot pro acceptance a soak ověření

Bezpečný default konfigurace je:

- `audio_aec_mode=windows_system_aec`
- `audio_device_mode=auto`
- `audio_session_profile=production`
- `audio_diagnostics_enabled=false`

Diagnostické přepínače jsou oddělené od produkčních rozhodovacích cest. `custom_lab` je explicitní laboratorní mód, ne běžná produkční cesta.

## Build pro macOS

Pro vytvoření `.app` balíčku je v repu připravený PyInstaller build skript:

```bash
python3 tools/build_macos_app.py
```

Výstup najdeš v `dist/ChatbotKaja.app`.

Build používá `app_gui.py` jako vstupní bod, protože je to bezpečnější top-level script pro PyInstaller než balení `kajovochat/__main__.py`.
