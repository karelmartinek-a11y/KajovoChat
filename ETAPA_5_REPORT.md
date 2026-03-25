# ETAPA 5 REPORT

## Zadání etapy
Rozřezat historickou kompatibilní vrstvu `kajovochat/services/audio_service.py` do finálních modulů pod `kajovochat/audio/`, uzavřít ownership a zajistit, aby `services/` už nebylo druhou audio architekturou.

## Změněné soubory
- `kajovochat/audio/devices.py` — nový finální modul pro discovery, device heuristiky, fingerprint a kalibraci.
- `kajovochat/audio/dsp_helpers.py` — nový finální modul pro reference/capture/render DSP helpery a adaptivní AEC utility.
- `kajovochat/audio/io/__init__.py` — nový lazy export vstupní bod pro audio I/O vrstvu.
- `kajovochat/audio/io/common.py` — nový modul pro shared I/O dataclasses a resampling helper.
- `kajovochat/audio/io/runtime.py` — nový finální modul pro `AudioRecorder`, `AudioPlayer`, `RealtimeMicStream`, `DuplexAudioSession`, `VADMonitor`.
- `kajovochat/audio/device_graph.py` — importy přepojené na `audio.io`.
- `kajovochat/audio/runtime_resources.py` — importy přepojené na `audio.io`.
- `kajovochat/audio/session_policy.py` — importy přepojené na `audio.devices` a `audio.io`.
- `kajovochat/audio/session_manager.py` — importy přepojené na `audio.devices` a `audio.io`.
- `kajovochat/main.py` — importy přepojené na nové finální audio moduly.
- `kajovochat/tools/voice_preview.py` — import `AudioPlayer` přepojený na `audio.io`.
- `kajovochat/services/audio_service.py` — zredukováno na čistou kompatibilní re-export vrstvu bez logiky.
- `kajovochat/services/audio_session_manager.py` — kompatibilní import `DuplexAudioSession` přepojený na `audio.io`.
- `tests/test_audio_contracts.py` — test importů přepojený na `audio.io`.
- `tests/test_audio_player_reference.py` — test importů přepojený na `audio.io`.
- `tests/test_guard_services.py` — test DSP importů přepojený na `audio.dsp_helpers`.
- `tests/test_main_audio_guard.py` — testy importů a monkeypatch cílů přepojené na `audio.devices`.
- `tests/test_audio_service_compat.py` — nový test pro čistou kompatibilní vrstvu bez logiky.

## Jaké odpovědnosti byly vytaženy z `audio_service.py`
### Přesunuto do `kajovochat/audio/io/`
- `DuplexAudioSession`
- `RealtimeMicStream`
- `AudioPlayer`
- `AudioRecorder`
- `VADMonitor`
- `CapturedAudioChunk`
- `RecordResult`
- `_resample_pcm16_mono`

### Přesunuto do `kajovochat/audio/devices.py`
- `AudioCalibrationResult`
- `list_audio_devices()`
- `pick_audio_device()`
- `format_device_help()`
- `build_device_fingerprint()`
- `calibrate_audio_devices()`
- `calibrate_audio_devices_advanced()`
- device name / audio mode / discovery helpery

### Přesunuto do `kajovochat/audio/dsp_helpers.py`
- `ReferencePrepResult`
- `_rms()`
- `_normalized_similarity()`
- `_extract_reference_segment()`
- `_extract_reference_context()`
- `_candidate_shifts()`
- `_find_best_alignment()`
- `_find_best_alignment_exhaustive()`
- `_WebRTCAECBackend`
- `AdaptiveEchoCanceller`
- `suppress_echo_from_pcm16()`

## Jaké nové finální moduly vznikly
- `kajovochat/audio/devices.py`
- `kajovochat/audio/dsp_helpers.py`
- `kajovochat/audio/io/common.py`
- `kajovochat/audio/io/runtime.py`
- `kajovochat/audio/io/__init__.py`

Tím je finální ownership rozdělen takto:
- `audio/io/*` = runtime I/O ownership
- `audio/devices.py` = discovery + calibration + device identity
- `audio/dsp_helpers.py` = úzce vymezené DSP a reference helpery

## Které legacy importy zůstaly a proč jsou čistě bez logiky
### `kajovochat/services/audio_service.py`
Zůstává jen kvůli kompatibilním importům mimo repo nebo pro postupné odstranění starších callsite. Soubor už obsahuje jen:
- `from ... import ...` re-exporty z `audio/`
- `__all__`

Neobsahuje:
- žádnou třídu
- žádnou funkci
- žádný runtime ownership
- žádné device/DSP/audio rozhodování

### `kajovochat/services/audio_session_manager.py`
Zůstává jako tenká kompatibilní re-export vrstva nad `kajovochat/audio/session_manager.py`. `DuplexAudioSession` už bere z `audio.io`, ne z `services/audio_service.py`.

## Jak bylo ověřeno, že `services/` už není druhá architektura
1. Produkční importy byly přepojeny na nové moduly:
   - `main.py` už bere audio symboly z `audio.devices`, `audio.dsp_helpers`, `audio.io`
   - `audio/session_manager.py`, `audio/session_policy.py`, `audio/runtime_resources.py`, `audio/device_graph.py` už neimportují runtime audio z `services/audio_service.py`
2. Grep kontrola po změně ukazuje, že reference na `audio_service` zůstaly jen v novém kompatibilním testu.
3. `tests/test_audio_service_compat.py` AST kontroluje, že `kajovochat/services/audio_service.py` je tvořen pouze `ImportFrom` a `Assign`, tedy čistý re-export bez logiky.
4. `tests/test_audio_service_compat.py` zároveň ověřuje, že kompatibilní symboly skutečně ukazují na nové finální moduly.

## Upravované a nové testy
Upravené:
- `tests/test_audio_contracts.py`
- `tests/test_audio_player_reference.py`
- `tests/test_guard_services.py`
- `tests/test_main_audio_guard.py`

Nové:
- `tests/test_audio_service_compat.py`

Nová coverage:
- kompatibilní `audio_service.py` je čistý re-export bez top-level logiky
- kompatibilní exporty skutečně míří na `audio.devices` / `audio.io`

## Spuštěné kontroly
- `python -m compileall -q kajovochat app_gui.py tests`
- `pytest -q`

Výsledek:
- `142 passed, 2 skipped`

## Zbylé riziko
- Dokumentace v repu ještě může místy historicky zmiňovat `kajovochat/services/audio_service.py` jako implementační detail. Runtime ownership už tam ale není.
- Pokud nějaký externí neotestovaný integrátor mimo tento repozitář importuje privátní helpery ze staré cesty, bude stále fungovat přes re-exporty, ale cílová struktura je už `kajovochat/audio/*`.
