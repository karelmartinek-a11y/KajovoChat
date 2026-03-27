# Voice upgrade notes

Tento patch upravuje runtime směrem blíž k modernímu voice UX přes OpenAI Realtime API.

## Co je změněno

- hands-free relace nově startuje s `semantic_vad` místo čistého `server_vad`
- vstupní přepis je přepnutý z `whisper-1` na `gpt-4o-mini-transcribe`
- při barge-in se před lokálním zastavením přehrávání posílá `conversation.item.truncate`, aby kontext na serveru lépe odpovídal tomu, co uživatel skutečně slyšel
- testy pro Realtime klient dostaly ověření `semantic_vad` session update a truncate logiky

## Kde jsou změny v kódu

- `kajovochat/services/realtime_service.py`
- `kajovochat/audio/session_manager.py`
- `kajovochat/audio/transport_bridge.py`
- `kajovochat/audio/recovery.py`
- doplněné a upravené testy v `tests/`

## Ověření v tomto patchi

Proběhlo:

- `python -m compileall -q kajovochat app_gui.py tests`
- cílené testy:
  - `tests/test_realtime_service.py`
  - `tests/test_audio_session_manager.py`
  - `tests/test_recovery_supervisor_unit.py`
  - `tests/test_audio_architecture_unit_layers.py`
  - `tests/test_main_audio_guard.py`

Tyto cílené testy v patchi prošly.

## Poznámka k plné sadě testů

Plná `pytest -q` sada v linuxovém kontejneru neprošla celá, protože část acceptance testů je navázaná na prostředí a Windows audio backendy / evidence workflow. To není chyba syntaxe patchovaného runtime, ale omezení testovacího prostředí mimo cílový notebookový deployment.
