# ETAPA 3 REPORT

## Zadání etapy
Centralizovat veškerou hlasovou UX politiku do `VoiceGate` tak, aby capture gate, barge-in, TTS hold-off, echo-drop a reference gating měly jediný source of truth mimo UI a mimo runtime smyčky.

## Změněné soubory
- `kajovochat/audio/voice_gate.py`
- `kajovochat/audio/session_manager.py`
- `kajovochat/audio/session_runtime.py`
- `kajovochat/audio/bootstrap.py`
- `kajovochat/audio/session_policy.py`
- `kajovochat/audio/worker_controls.py`
- `kajovochat/main.py`
- `tests/test_main_audio_guard.py`

## Jaká UX pravidla byla přesunuta do VoiceGate
Do `kajovochat/audio/voice_gate.py` byla sjednocena tato rozhodnutí a runtime state:
- capture pass/drop rozhodnutí přes `VoiceGate.evaluate_capture_gate(...)`
- barge-in candidate a confirmed logika včetně streak pravidel a TTS hold-off vlivu
- TTS start hold a tail hold okna přes `VoiceGate.note_tts_window(...)`
- playback reference arm/warmup/mic suppression window přes `VoiceGate.update_playback_reference_state(...)`
- reference source selection přes `VoiceGate.select_reference_source(...)`
  - `live`
  - `cached`
  - `cached_tail`
  - `system`
  - `none`
- gate side-effect counters (`echo_drop_count`, `barge_in_chunk_count`) a jejich log throttle
- runtime snapshot pro observer/testy přes `VoiceGate.snapshot(...)`

Nové explicitní rozhodovací objekty:
- `VoiceGateThresholds`
- `ReferenceSelectionDecision`
- `GateDecision`
- `GateSideEffects`
- `VoiceGateSnapshot`

## Odkud byla logika odstraněna
### `kajovochat/audio/session_runtime.py`
Odstraněna vlastní konkurenční hlasová politika runtime smyčky:
- ruční reference ready / cached fallback větev
- přímé rozhodování o capture gate přes injektovaný callback
- separátní inkrementace gate side-effectů mimo VoiceGate

Runtime smyčka nyní jen:
- získá audio/AEC pozorování
- požádá `session_manager.select_reference_source(...)`
- požádá `session_manager.evaluate_capture_gate(...)`
- aplikuje vrácené rozhodnutí

### `kajovochat/audio/session_policy.py`
Odstraněny reference gating helpery:
- `is_reference_ready(...)`
- `can_use_cached_reference(...)`

`session_policy.py` už neobsahuje konkurenční gate/reference rozhodování.

### `kajovochat/main.py`
Odstraněny worker/main proxy bridge vrstvy do VoiceGate runtime:
- `_mic_suppressed_until`
- `_echo_drop_count`
- `_barge_in_chunk_count`
- `_last_echo_drop_reported`
- `_last_barge_in_reported`
- `_last_aec_diag_log_at`
- `_last_aec_success_log_at`
- `_playback_reference_armed`
- `_reference_warmup_until`
- `_cached_echo_reference`
- `_cached_reference_at`
- `_is_reference_ready(...)`
- `_can_use_cached_reference(...)`
- lokální wrapper `main.py` pro capture gate politiku

`main.py` už drží jen UI, wiring a prezentaci guard diagnostiky. V hlasové UX politice není source of truth.

### `kajovochat/audio/worker_controls.py`
Odstraněna duplicitní worker-local stopa `_awaiting_transcript`.

## Jak nyní VoiceGate vystavuje rozhodnutí
Přes `AudioSessionManager` jsou oficiálně dostupné jen tyto session-oriented vstupy do VoiceGate:
- `select_reference_source(...)`
- `evaluate_capture_gate(...)`
- `note_playback_activity(...)`
- `note_tts_rendering(...)`
- `voice_gate_snapshot(...)`
- `should_log_problem_diag(...)`
- `should_log_success_diag(...)`
- `note_diag_logged(...)`

`AudioSessionManager` už nevymýšlí paralelní hlasovou politiku; pouze čte a aplikuje rozhodnutí `VoiceGate`.

## Které testy byly přidány/upraveny
Upravené testy:
- `tests/test_main_audio_guard.py`
  - přesměrovány assertiony z worker/main proxy helperů na `VoiceGate` / `session_manager`
  - nové coverage pro:
    - `select_reference_source(...)` s live referencí
    - `select_reference_source(...)` s cached fallbackem
    - `voice_gate_snapshot(...)` jako centrální runtime pohled

Existující testy, které stále validují centralizovaný model:
- `tests/test_voice_gate_runtime.py`
  - potvrzují playback warmup, cached reference, TTS hold-off, barge-in streak a gate side-effects přímo ve `voice_gate.py`

## Jak bylo ověřeno, že VoiceGate je jediný source of truth
Architektonická kontrola po implementaci:
- `session_runtime.py` volá `session_manager.select_reference_source(...)` a `session_manager.evaluate_capture_gate(...)`
- `main.py` už neobsahuje reference/gate/barged-in runtime proxy stav ani helpery
- `session_callbacks.py` neobsahuje žádnou vlastní gate UX logiku
- `session_policy.py` už neobsahuje reference gating rozhodování

Kontrolní příkazy:
- `python -m compileall -q kajovochat app_gui.py`
- `pytest -q`

Výsledek:
- `138 passed, 2 skipped`

## Zbylé riziko
- Plný `pip install -r requirements.txt` v tomto prostředí neprošel kvůli buildu volitelného balíčku `aec-audio-processing`. Pro ověření byly doinstalovány jen potřebné runtime/test závislosti a aplikační testy i compile check proběhly úspěšně.
- `backend_aware_aec_metrics(...)` zůstává mimo `VoiceGate` rozhodovací proud jen jako normalizace AEC metrik pro vstup do VoiceGate; není to samostatná hlasová UX politika ani paralelní state machine.
