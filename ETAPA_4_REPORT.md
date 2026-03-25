# ETAPA 4 REPORT

## Zadání etapy
Centralizovat recovery rozhodování do `RecoverySupervisor` a udělat z `AudioTelemetry` jediný source of truth pro session health, backend health, fallback reason a kompletní recovery story.

## Změněné soubory
- `kajovochat/audio/telemetry.py`
- `kajovochat/audio/recovery.py`
- `kajovochat/audio/session_manager.py`
- `kajovochat/audio/transport_bridge.py`
- `kajovochat/audio/session_callbacks.py`
- `tests/test_audio_session_manager.py`

## Které recovery scénáře jsou teď centralizované
Do `kajovochat/audio/recovery.py` byly sjednoceny všechny provozní recovery decision paths:

- `transport_disconnect`
- `transport_timeout`
- `playback_stagnation`
- `reference_pipeline_unhealthy`
- `windows_system_aec_unhealthy`
- `windows_system_aec_unavailable`
- `webrtc_apm_unavailable` jako součást probe/fallback story
- `device_unavailable` z eskalace XRUN a device resetů
- `recovery_exhausted`

`RecoverySupervisor` teď rozhoduje o:
- transport reconnectu,
- backend fallbacku,
- potlačení oscilace,
- vyčerpání recovery,
- cílovém stavu relace (`recovering` / `failed`),
- cooldownu a záznamu recovery story.

## Jaké telemetrické údaje jsou nově centralizované
Do `kajovochat/audio/telemetry.py` byly doplněny a centralizovány tyto session-level údaje:

- `requested_backend`
- `selected_backend`
- `fallback_chain_step`
- `fallback_reason`
- `degradation_cause`
- `reconnect_attempts`
- `recovery_attempts_total`
- `recovery_successes_total`
- `recovery_failures_total`
- `xrun_events_total`
- `device_resets_total`
- `reference_health_timeline`
- `backend_health_score`
- `session_start / probe_start / probe_done / active / stop` timing
- turn timing agregace (`avg_first_audio_latency_ms`, `avg_response_latency_ms`, maxima, počty dokončených odpovědí)
- runtime backlog / playback progress snapshot pro watchdog
- serializovatelná `recovery_story`

Nové explicitní serializovatelné objekty:
- `ReferenceHealthEvent`
- `RecoveryStoryEvent`
- `SessionTelemetrySnapshot`

## Odkud byla rozhodovací logika odstraněna
### `kajovochat/audio/session_manager.py`
Ze session manageru byla odstraněna vlastní recovery politika:
- reference-health fallback rozhodování
- windows native AEC unhealthy fallback rozhodování
- watchdog rozhodování pro playback stagnation
- watchdog rozhodování pro realtime idle timeout
- transport error klasifikace jako hlavní decision center

`AudioSessionManager` teď už jen:
- deleguje health observations do `RecoverySupervisor`
- aplikuje backend switch přes `_attempt_backend_fallback(...)`
- mění session state podle rozhodnutí recovery vrstvy
- vystavuje log payloady a session entry pointy

### `kajovochat/audio/session_callbacks.py`
Callback vrstva zůstala čistě delegační. Nepřidává žádnou reconnect policy ani vlastní health/fallback truth source.

### `kajovochat/audio/transport_bridge.py`
Transport vrstva zůstává pouze transportní. Byla doplněna o čistý `connection_health_snapshot()` a `is_connected`, ale neobsahuje reconnect rozhodování.

## Jak je řešena anti-oscillation ochrana
Anti-oscillation guard je centralizovaný v `RecoverySupervisor`:

- fallback cooldown přes `_last_fallback_reason` + `_last_fallback_at`
- reconnect cooldown přes `_last_reconnect_reason` + `_last_reconnect_at`
- scheduled reconnect guard přes `telemetry.scheduled_reconnect_at`
- deterministická policy mapa `failure_reason -> action -> target_state -> cooldown`

Důsledek:
- backend fallback se nespouští opakovaně každých pár bloků pro stejný důvod,
- reconnect se neplánuje paralelně vícekrát,
- reconnect transportu a backend fallback jsou oddělené jevy s oddělenou story.

## Jak nyní log/snapshot vypráví recovery story
`AudioTelemetry.serializable_snapshot(...)` vrací serializovatelný snapshot se sekcemi:
- `timings`
- `turn_latency`
- `health`
- `reference_health_timeline`
- `recovery_story`

`AudioSessionManager._build_log_payload(...)` teď do log payloadu přidává:
- `session_telemetry_snapshot`
- `transport_health`

Tím lze z jednoho snapshotu nebo log záznamu rekonstruovat:
- jaký backend byl požadovaný a jaký byl zvolen,
- proč došlo k fallbacku,
- jaký byl reconnect/fallback chain step,
- jaké bylo reference zdraví v čase,
- kdy probe/start/active/stop skutečně proběhly,
- jak dopadly recovery pokusy.

## Jaké testy a fault injection scénáře byly doplněny nebo upraveny
Upravené testy:
- `tests/test_audio_session_manager.py`

Doplněné coverage:
- serializovatelný telemetry snapshot obsahuje `reference_health_timeline` a `recovery_story`
- runtime watchdog timeout vede přes centralizovaný `RecoverySupervisor` na `transport_reconnect`

Stávající testy, které nyní validují finální centralizaci:
- controlled reference fallback
- windows AEC unhealthy fallback
- xrun / device reset telemetrie
- recovery exhaustion logování
- backend switch a degraded transition counters
- transport timeout jako recoverable session failure

## Ověření
Spuštěné kontroly:
- `python -m compileall -q kajovochat app_gui.py`
- `pytest -q`

Výsledek:
- `140 passed, 2 skipped`

## Zbylé riziko
- Plná instalace všech závislostí z `requirements.txt` může v některých prostředích znovu narazit na build volitelného balíčku `aec-audio-processing`. Samotná stage 4 implementace byla ověřena compile checkem a kompletním test během v tomto repu.
- `AecEngine` stále zůstává canonical výběrovou vrstvou backend chainu při startovním probingu. Recovery rozhodování po startu už ale běží pouze přes `RecoverySupervisor` a provozní health truth source je pouze `AudioTelemetry`.
