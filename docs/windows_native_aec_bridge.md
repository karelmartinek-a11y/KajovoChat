# Windows native AEC bridge

Datum: 2026-03-25

## Cíl

Držet `windows_system_aec` jako první produkční volbu pro notebookový audio stack na Windows a současně mít auditovatelný fallback chain bez návratu k původnímu hybridnímu chaosu.

## Aktuální produkční policy

Kanonický produkční chain je nyní:

1. `windows_system_aec`
2. `webrtc_apm`
3. `degraded_no_aec`

`custom_lab` zůstává laboratorní režim a není součástí produkční notebookové cesty.

## Co je v repozitáři

Python bridge a probe vrstva:

- [`C:\GitHub\KajovoChat\kajovochat\services\windows_native_aec.py`](C:\GitHub\KajovoChat\kajovochat\services\windows_native_aec.py)
- detekce helperu přes `KAJOVOCHAT_WINDOWS_AEC_DLL` a `KAJOVOCHAT_WINDOWS_APO_DLL`
- detekce nainstalovaného APO driveru přes `pnputil`
- session-level použití přes `AudioSessionManager` a `WindowsSystemAecBackendRunner`

Nativní helpery:

- [`C:\GitHub\KajovoChat\native\windows_aec_helper\`](C:\GitHub\KajovoChat\native\windows_aec_helper\)
- [`C:\GitHub\KajovoChat\native\windows_apo_helper\`](C:\GitHub\KajovoChat\native\windows_apo_helper\)

## Aktuální nativní kontrakty

Repo drží dvě nativní cesty:

- `kajovochat_aec_*`
  - reference-based helper kontrakt pro klasický native AEC běh
- `kajovochat_apo_capture_*`
  - capture-only kontrakt pro APO větev bez app-level playback reference

Python bridge při dostupném APO driveru preferuje `kajovochat_apo_capture_*`. Pokud capture symboly nejsou k dispozici, spadne se na `kajovochat_aec_*`.

## Session-oriented Python API

Bridge už nevystavuje jen per-frame `process(...)`, ale i session kontrakt:

- `WindowsNativeAECSessionConfig`
- `WindowsNativeAECSession.start()`
- `WindowsNativeAECSession.write_render_frame(frame)`
- `WindowsNativeAECSession.submit_capture_frame(raw_mic_pcm16, mono_ns, stream_delay_ms, render_ref_pcm16=None)`
- `WindowsNativeAECSession.read_capture_frame(timeout_ms)`
- `WindowsNativeAECSession.get_health_snapshot()`
- `WindowsNativeAECSession.stop()`
- `WindowsNativeAECSession.close()`

Tento kontrakt vrací explicitní [`CaptureFrame`](C:\GitHub\KajovoChat\kajovochat\audio\contracts.py) a health snapshot místo toho, aby produkční `windows_system_aec` sahal přímo na holý buffer procesor.

## Očekávané ABI helperu

Reference-based helper:

- `kajovochat_aec_create(int samplerate, int filter_length, int max_shift_samples) -> void*`
- `kajovochat_aec_destroy(void* handle)`
- `kajovochat_aec_process(void* handle, mic, mic_samples, reference, reference_samples, delay_ms, out_pcm, out_capacity, out_quality, out_improvement, out_residual, out_is_strong) -> int`

APO capture kontrakt:

- `kajovochat_apo_capture_create(int samplerate) -> void*`
- `kajovochat_apo_capture_destroy(void* handle)`
- `kajovochat_apo_capture_process(void* handle, mic, mic_samples, out_pcm, out_capacity, out_quality, out_voice_likelihood, out_processing_flags) -> int`

## Jak to číst v logu

Pro audit jedné relace sleduj hlavně:

- `audio_backend_selected`
- `audio_backend_fallback`
- `audio_reference_health`
- `audio_session_state`
- `reconnect_scheduled`
- `reconnect_ok`
- `recovery_exhausted`
- `aec_diag`

Kanonické rozhodnutí o backendu je v session telemetrii přes:

- `requested_backend`
- `selected_backend`
- `fallback_reason`
- `degradation_cause`

`aec_diag` je block-level diagnostika, ne session-level zdroj pravdy.

## Poctivý stav

Bridge je dnes výrazně dál než původní per-buffer experiment:

- má session-oriented Python API
- umí použít APO capture kontrakt
- `windows_system_aec` je napojený jako produkční backend

Pořád ale platí, že jde o helper-backed systémovou cestu. Pokud bude cílem úplně čistý systémový backend bez mezivrstvy, další krok už bude hlubší nativní integrace mimo současný DLL bridge.

## Build požadavky

Pro build helperů je potřeba:

- CMake
- MSVC / Visual Studio Build Tools
- Windows SDK
- pro release packaging také `Inf2Cat` a `signtool`
