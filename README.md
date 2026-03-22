# Chatbot Kája

Desktop aplikace v `PySide6` pro hlasovou konverzaci přes OpenAI Realtime API.

## Co aplikace umí

- hands-free hlasovou konverzaci přes klik na orb asistenta
- push-to-talk přes tlačítko zeměkoule
- shader-based living orb renderer s audio-reaktivní animací podle skutečně přehrávaného audia
- barge-in během mluvení asistenta
- softwarové tlumení self-hearing na notebookových reproduktorech

## Požadavky

- Python `3.11+`
- funkční mikrofon a reproduktory nebo sluchátka
- platný OpenAI API klíč
- GPU/driver s podporou OpenGL pro plnou living orb cestu

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

Po spuštění otevřete dialog `OpenAI` a vložte API klíč. Klíč se ve Windows ukládá přes DPAPI, na ostatních platformách přes systémový keyring, pokud je dostupný.

## Konfigurace

Konfigurace se ukládá do `settings.json` v uživatelském profilu aplikace. Technická dokumentace living orbu je v [docs/living_orb_renderer.md](docs/living_orb_renderer.md).

## Struktura projektu

- `kajovochat/` hlavní aplikační balíček
- `kajovochat/main.py` GUI a orchestrace hlasové relace
- `kajovochat/settings.py` minimální produktové nastavení a systémový prompt
- `kajovochat/services/` audio, realtime websocket, OpenAI služby a logování
- `kajovochat/widgets/` vlastní UI widgety
- `kajovochat/orb/` GPU living orb engine, audio analýza, stavový blending a shader renderer
- `kajovochat/resources/assets/` obrázkové assety
- `tests/` testy

## Test a kontrola

```bash
python -m compileall -q kajovochat app_gui.py
pytest -q
```

## Build pro macOS

Pro vytvoření `.app` balíčku je v repu připravený PyInstaller build skript:

```bash
python3 tools/build_macos_app.py
```

Výstup najdeš v `dist/ChatbotKaja.app`.

Build používá `app_gui.py` jako vstupní bod, protože je to bezpečnější top-level script pro PyInstaller než balení `kajovochat/__main__.py`.
