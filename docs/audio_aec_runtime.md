# Audio AEC Runtime

## Produkční backend policy

Produkční relace už nepoužívá neřízený hybrid několika AEC strategií. Session policy je pevná:

1. `windows_system_aec`
2. `webrtc_apm`
3. `degraded_no_aec`

`custom_lab` je pouze laboratorní režim. Slouží pro DSP experimenty, regresní testy a analýzu reference pipeline, ne jako produkční default.

## Kde se co rozhoduje

- `AudioSessionManager` drží session policy, backend probing a lifecycle
- `AecEngine` drží backend chain pro zvolený policy mode
- `AdaptiveEchoCanceller` provádí block-level DSP čištění a guard diagnostiku
- `RecoverySupervisor` řídí reconnect, fallback a anti-oscillation guard

## Session logy

Hlavním zdrojem jsou session `.jsonl` logy. Pro audio diagnostiku sleduj hlavně:

- `audio_session_state`
- `audio_backend_selected`
- `audio_backend_fallback`
- `audio_reference_health`
- `aec_diag`
- `aec_summary`
- `reconnect_scheduled`
- `reconnect_ok`
- `reconnect_failed`
- `echo_guard`

## Důležité telemetrické položky

- `requested_backend`
- `selected_backend`
- `fallback_reason`
- `device_fingerprint`
- `session_timing`
- `reference.ready`
- `reference.health`
- `degradation_cause`
- `recovery_attempts`

## Interpretace

`selected_backend=windows_system_aec` znamená, že relace běží v preferovaném produkčním režimu. Pokud není systémový backend dostupný nebo reference pipeline dlouhodobě selhává, relace přejde řízeně na `webrtc_apm`, případně až na `degraded_no_aec`.

`backend=custom` v `aec_diag` se má objevovat pouze v explicitním `custom_lab` režimu. Produkční relace se auditují přes session logy: `requested_backend`, `selected_backend`, `fallback_reason`, `degradation_cause` a `backend_chain`.

## Praktický postup ladění

1. Otevři `.jsonl` log jedné relace.
2. Najdi první `audio_backend_selected` a ověř `requested_backend` vs. `selected_backend`.
3. Pokud došlo k propadu, hledej `audio_backend_fallback` a `fallback_reason`.
4. Zkontroluj `audio_reference_health` a `aec_diag`, zda byla reference skutečně ready.
5. Teprve potom vyhodnocuj `aec_summary`.

Tak je možné z jediné relace jednoznačně rekonstruovat, co se stalo: jaký backend byl požadován, který byl skutečně aktivní, proč došlo k fallbacku a zda šlo o degradaci kvůli reference pipeline nebo kvůli transportu.
