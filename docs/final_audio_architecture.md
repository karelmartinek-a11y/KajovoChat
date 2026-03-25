# Finální stav audio architektury

Repo po etapě 8 drží finální cílovou audio architekturu bez druhého source of truth v produkční větvi.

## Jediné produkční autority

- `AudioSessionManager` je jediný session entry point a aplikační vrstva pro audio relaci.
- `VoiceGate` je jediný source of truth pro hlasovou UX politiku: capture gate, reference gating, TTS hold-off a barge-in potvrzení.
- `RecoverySupervisor` je jediný source of truth pro reconnect a backend fallback policy.
- `AudioTelemetry` je jediný source of truth pro session health, fallback story, recovery story a serializovatelný snapshot.
- `windows_system_aec` je finální helper-backed produkční backend detail schovaný za session-level kontraktem `kajovochat.audio.windows_system_aec`.

## Produkční backend chain

1. `windows_system_aec`
2. `webrtc_apm`
3. `degraded_no_aec`

Pro headset topologii se chain neřeší přes AEC fallback, ale přepíná se do explicitního first-class režimu `headset_clean`.

## Důkazy

- acceptance scénáře: 5
- integration scénáře: 7
- soak/fault-injection scénáře: 5
- tabulka verdictů: `FINAL_ACCEPTANCE_MATRIX.md`
- evidence soubory: `docs/audio_acceptance_evidence/*`
- scripted harness: `tools/audio_architecture_harness.py`
- generátor evidence: `tools/generate_audio_acceptance_evidence.py`

## Poznámka k HW závislosti

Plně reálné notebook/headset chování vyžaduje fyzický hardware a konkrétní Windows audio topologii. CI proto používá deterministické scripted runs a fault injection, ale produkční rozhodovací logika zůstává stejná jako v aplikaci.
