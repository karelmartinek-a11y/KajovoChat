# Finální stav audio architektury

Tento dokument uzavírá finální audio architekturu po etapě 8. Produkční audio stack má jediný session lifecycle model, jediný VoiceGate pro hlasovou UX politiku, jediný RecoverySupervisor pro recovery rozhodování a jedinou AudioTelemetry pro provozní příběh relace.

## Produkční ownership

- `AudioSessionManager` drží session lifecycle a aplikuje rozhodnutí podřízených vrstev.
- `VoiceGate` je jediný source of truth pro capture gate, reference gating, TTS hold-off a barge-in potvrzení.
- `RecoverySupervisor` je jediný source of truth pro reconnect a backend fallback policy.
- `AudioTelemetry` je jediný source of truth pro session health, fallback/recovery story a serializovatelný snapshot.
- `windows_system_aec` je finální helper-backed produkční backend detail uzavřený za session-level kontraktem `kajovochat.audio.windows_system_aec`.

## Produkční backend chain

1. `windows_system_aec`
2. `webrtc_apm`
3. `degraded_no_aec`

Topologie `wired_headset` nebo ekvivalent přepíná do explicitního first-class režimu `headset_clean`.

## Důkazy

- acceptance matrix: `FINAL_ACCEPTANCE_MATRIX.md`
- acceptance evidence: `docs/audio_evidence/acceptance/*.json`
- integration evidence: `docs/audio_evidence/integration/*.json`
- soak evidence: `docs/audio_evidence/soak/*.json`
- scripted harness: `tests/audio_acceptance_harness.py` + `tools/generate_audio_acceptance_evidence.py`
- unit/integration/soak testy: `pytest -q` nebo cílené běhy přes nové test files
