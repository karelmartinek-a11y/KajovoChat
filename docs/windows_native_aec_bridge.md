# Windows System AEC backend

Datum: 2026-03-25

## Definitivní varianta

`windows_system_aec` je v tomto repozitáři uzavřený jako **finální helper-backed produkční backend detail**. Nejde o dočasný bridge, ale o stabilní implementační variantu schovanou za session-level veřejným kontraktem v `kajovochat.audio.windows_system_aec`.

`AudioSessionManager` nezná helper symboly, názvy DLL ani ABI detaily. Zná jen veřejný kontrakt:

- `probe_windows_system_aec()`
- `windows_system_aec_healthcheck()`
- canonical backend name `windows_system_aec`

## Produkční backend chain

1. `windows_system_aec`
2. `webrtc_apm`
3. `degraded_no_aec`

Pro headset topologii se chain neřeší přes AEC fallback, ale přepíná se do first-class režimu `headset_clean`.

## Skryté helper detaily

Nízkourovňová helper implementace zůstává v `kajovochat/services/windows_native_aec.py`, ale je to čistě interní detail finálního backendu. Session a recovery vrstvy pracují výhradně se session-level healthcheckem a s canonical backend názvy.

Interní helper detail může podle dostupnosti použít dvě nativní cesty:

- `kajovochat_aec_*` pro render-reference kontrakt
- `kajovochat_apo_capture_*` pro systémový capture kontrakt s nainstalovaným APO driverem

Tato volba je interní. Není to druhý veřejný backend ani druhý source of truth.

## Audit a evidence

Pro audit jedné relace sleduj hlavně:

- `audio_backend_selected`
- `audio_backend_fallback`
- `audio_session_state`
- `audio_reference_health`
- `reconnect_scheduled`
- `reconnect_ok`
- `recovery_exhausted`
- `session_telemetry_snapshot`

`aec_diag` zůstává block-level diagnostika. Produkční truth source jsou `AudioTelemetry` a `RecoverySupervisor`.

## Build požadavky

Pro build helperů je potřeba:

- CMake
- MSVC / Visual Studio Build Tools
- Windows SDK
- pro release packaging také `Inf2Cat` a `signtool`
