# ETAPA 7 REPORT

## Zvolená finální varianta
Zvolil jsem **definitivně helper-backed implementaci jako finální produkční backend detail** pro `windows_system_aec`.

Nejde už o neuzavřený bridge ani dočasný workaround. Veřejný session-oriented kontrakt je uzavřen v modulu `kajovochat/audio/windows_system_aec.py`, zatímco nízkoúrovňová DLL/APO implementace zůstává skrytá v `kajovochat/services/windows_native_aec.py` jako interní implementační detail.

## Proč je tato varianta definitivní
- Repo už reálně stojí na helper-backed native implementaci a na APO/system-capture variantě.
- Úplná náhrada čistým OS backendem bez helperu by v tomto repu nebyla realistická bez zásahu do nativní části mimo Python vrstvu.
- Místo nejasné mezivrstvy jsem helper-backed cestu formalizoval jako **finální backend implementation detail** se stabilním veřejným kontraktem `windows_system_aec`.
- `AudioSessionManager` už nečte helper probe přímo a nezná DLL/APO rozhodovací detaily.

## Jak jsou schované nebo odstraněné helper detaily
Do veřejné session vrstvy jsem zavedl nový modul:
- `kajovochat/audio/windows_system_aec.py`

Ten poskytuje:
- `WindowsSystemAecProbe`
- `WindowsSystemAecHealthcheck`
- `probe_windows_system_aec()`
- `windows_system_aec_healthcheck()`

Tím je session-oriented vrstva oddělena od nízkoúrovňových detailů v `services/windows_native_aec.py`.

Konkrétní změny:
- `kajovochat/audio/session_manager.py` už používá pouze `windows_system_aec_healthcheck()`.
- `kajovochat/main.py` a `kajovochat/settings.py` už používají veřejný probe `probe_windows_system_aec()` místo přímého čtení native helper modulu.
- veřejné backend označení je sjednocené na `windows_system_aec`; v produkční session vrstvě už se backend nepropaguje jako `windows_native` ani `windows_system_capture`.

## Jak je zajištěn fallback chain bez změny veřejných kontraktů
Fallback chain zůstal deterministický a auditovatelný:
- `windows_system_aec -> webrtc_apm -> degraded_no_aec`

To je zachované přes stejné veřejné session kontrakty:
- `AecEngine.select_backend(...)`
- jednotný `windows_system_aec_healthcheck()`
- `AudioSessionManager._probe_windows_system_aec()` vrací vždy session-level stav, ne helper detail
- `RecoverySupervisor` dál rozhoduje podle session-level failure reason (`windows_system_aec_unhealthy`, `windows_system_aec_unavailable`, atd.)

`AudioSessionManager` nezná helper DLL, APO path ani helper-specific fallback rozhodování. Vidí jen:
- backend available / unavailable
- veřejný důvod
- stejný fallback kontrakt jako pro ostatní backendy

## Upravené soubory
- `kajovochat/audio/windows_system_aec.py`
  - nový veřejný session-oriented kontrakt pro `windows_system_aec`
- `kajovochat/audio/session_manager.py`
  - odstraněn přímý probe na helper modul, sjednocen healthcheck kontrakt
- `kajovochat/audio/recovery.py`
  - sjednocena interpretace `windows_system_aec` health observací bez helper názvosloví v decision path
- `kajovochat/audio/session_runtime.py`
  - sjednoceno backend jméno v runtime diagnostice a decision path
- `kajovochat/audio/voice_gate.py`
  - backend-aware metriky sjednoceny na `windows_system_aec`
- `kajovochat/audio/aec_backends/windows_system.py`
  - backend runner vrací veřejný backend `windows_system_aec`
- `kajovochat/audio/aec_backends/webrtc_apm.py`
  - preferenční logika přepnuta na veřejný backend kontrakt
- `kajovochat/audio/dsp_helpers.py`
  - system-capture větev vrací veřejný backend `windows_system_aec`
- `kajovochat/services/windows_native_aec.py`
  - formalizován jako nízkoúrovňová implementace finálního backendu, veřejný health snapshot sjednocen na `windows_system_aec`
- `kajovochat/main.py`
  - debug/UI probe používá veřejný audio kontrakt
- `kajovochat/settings.py`
  - promotion logic používá veřejný audio kontrakt
- `tests/test_audio_session_manager.py`
  - upraveny healthcheck monkeypatch body a očekávání veřejného backendu
- `tests/test_guard_services.py`
  - upravena očekávání na veřejný backend kontrakt `windows_system_aec`
- `tests/test_main_audio_guard.py`
  - upraven probe kontrakt pro debug guard payload
- `tests/test_settings.py`
  - přepnuto na nový veřejný probe modul
- `tests/test_windows_native_aec_session.py`
  - session kontrakt a health snapshot sjednoceny na `windows_system_aec`
- `tests/test_windows_system_aec_contract.py`
  - nový test veřejného kontraktu a healthcheck vrstvy

## Jaké testy pokrývají windows_system_aec
Přidané nebo upravené testy:
- `tests/test_windows_system_aec_contract.py`
  - ověřuje veřejný probe wrapper a session-level healthcheck kontrakt
- `tests/test_audio_session_manager.py`
  - ověřuje výběr backendu, fallback na `webrtc_apm`, degradaci a health decision path bez helper detailů v manageru
- `tests/test_windows_native_aec_session.py`
  - ověřuje, že session-oriented health snapshot a capture frame používají veřejný backend `windows_system_aec`
- `tests/test_guard_services.py`
  - ověřuje system-capture chování při sjednoceném backend jménu
- `tests/test_main_audio_guard.py`
  - ověřuje debug payload přes nový veřejný probe kontrakt
- `tests/test_settings.py`
  - ověřuje promotion logic přes nový veřejný probe kontrakt

## Spuštěné checky
- `python -m compileall -q kajovochat app_gui.py` → PASS
- `pytest -q` → PASS (`149 passed, 2 skipped`)

## Rizika / omezení
- Nativní DLL/APO implementace stále fyzicky existuje v `services/windows_native_aec.py`, protože repo na ní objektivně stojí. Není však už architektonicky nejasná: je to **nízkoúrovňový final backend implementation detail**, ne druhý veřejný backend model.
- Neprováděl jsem změny v C/C++ helper projektech pod `native/`; etapa 7 uzavírá Python produkční architekturu a kontrakty nad nimi.
