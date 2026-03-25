# ETAPA 6 REPORT

## Změněné soubory
- `kajovochat/audio/aec_engine.py`
- `kajovochat/audio/voice_gate.py`
- `kajovochat/audio/telemetry.py`
- `kajovochat/audio/contracts.py`
- `kajovochat/audio/session_manager.py`
- `kajovochat/audio/session_runtime.py`
- `tests/test_voice_gate_runtime.py`
- `tests/test_audio_session_manager.py`

## Jak je teď definovaný `degraded_no_aec`
`degraded_no_aec` už není jen poslední technická položka v backend chainu. V `aec_engine.py` má explicitní produktový kontrakt `AecProductMode` s těmito vlastnostmi:
- klíč režimu `notebook_builtin_degraded_no_aec`
- status `Nouzový režim bez AEC`
- UI/log text `Audio: nouzový režim degraded_no_aec bez AEC, s konzervativním capture gate.`
- capture gate policy `degraded_no_aec`
- bez požadavku na playback reference
- auditovatelný `degradation_reason`
- explicitní recovery policy `probe_richer_backend_again` jen pro notebookovou produkční topologii, jinak `stay_degraded`
- retry budget `2` pro návrat na bohatší backend, jinak `0`

Do `session_manager.py` jsem přidal propis tohoto kontraktu do captionů, logů a telemetrie. Přechod do `degraded_no_aec` se teď loguje s konkrétní příčinou a s produktovým statusem, ne jen změnou backend stringu.

## Jak je teď definovaný `headset_clean`
`headset_clean` je teď explicitní first-class režim s vlastním produktem kontraktem:
- klíč režimu `headset_clean`
- status `Headset clean`
- UI/log text `Audio: headset clean režim bez AEC a bez očekávání playback reference.`
- capture gate policy `headset_clean`
- žádná očekávaná playback reference
- recovery policy `topology_locked`
- retry budget `0`, protože nejde o degradaci notebookového AEC chainu, ale o cílovou headset topologii

To znamená, že headset cesta už není implicitní speciální případ odvozený bokem ze jména zařízení, ale samostatný produktový mód s dohledatelným kontraktem v kódu, logu i snapshotu telemetrie.

## Jaké nové telemetry a UX rozdíly byly zavedeny
V `telemetry.py` a `contracts.py` jsou nově centralizovaná pole:
- `product_mode_key`
- `product_status`
- `capture_gate_policy`
- `recovery_policy`

Tato pole se propisují do:
- `SessionTelemetrySnapshot`
- `BackendHealthSnapshot`
- `SessionHealth`
- session log payloadů z `session_manager.py`

Telemetrie teď jednoznačně rozlišuje minimálně tyto režimy:
- `notebook_builtin+windows_system_aec`
- `notebook_builtin+webrtc_apm`
- `notebook_builtin+degraded_no_aec`
- `headset_clean`

UX změny:
- `degraded_no_aec` má explicitní produktový caption s důvodem degradace a recovery politikou
- `headset_clean` má explicitní caption místo tichého implicitního zvláštního případu
- `session_runtime.py` už neřeší reference požadavek přes natvrdo vepsané backend stringy, ale čte ho přes produktový kontrakt session vrstvy

## Jak je změněná VoiceGate politika
V `voice_gate.py` jsem zavedl explicitní backend-aware capture gate policy:
- `degraded_no_aec` používá konzervativnější capture gate a přísnější barge-in potvrzení
- `headset_clean` vypíná echo-drop politiku, protože v této topologii není playback reference/AEC očekávaná produktová podmínka
- standardní notebookové AEC režimy zůstávají na běžné politice

Tím je chování těchto dvou režimů viditelné přímo v produkční rozhodovací vrstvě a ne jen v názvu zvoleného backendu.

## Jaké testy pokrývají vstup do režimů a jejich chování
Přidané nebo upravené testy:
- `tests/test_voice_gate_runtime.py::test_degraded_no_aec_policy_is_more_conservative_than_standard`
- `tests/test_voice_gate_runtime.py::test_headset_clean_policy_disables_echo_drop`
- `tests/test_audio_session_manager.py::test_aec_engine_product_mode_contracts_make_product_modes_explicit`
- `tests/test_audio_session_manager.py::test_session_manager_telemetry_exposes_product_mode_for_headset`
- `tests/test_audio_session_manager.py::test_session_manager_logs_degraded_reason_and_product_mode`

Pokrytí těchto testů:
- explicitní kontrakt obou produktových režimů
- rozdílná VoiceGate politika pro `degraded_no_aec` a `headset_clean`
- propagace product mode do telemetrie
- logování konkrétní příčiny přechodu do `degraded_no_aec`
- produktový status a recovery policy ve snapshotu a log payloadu

## Ověření
Spuštěné checky:
- `python -m compileall -q kajovochat app_gui.py`
- `pytest -q`

Výsledek:
- `147 passed, 2 skipped`
