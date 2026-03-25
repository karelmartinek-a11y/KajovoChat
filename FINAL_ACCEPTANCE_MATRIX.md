# FINAL_ACCEPTANCE_MATRIX

| scénář | způsob ověření | backend chain | expected final state | telemetry evidence | pass/fail |
|---|---|---|---|---|---|
| start_handsfree_session | pytest + scripted harness | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/integration/start_handsfree_session_snapshot.json` | **PASS** |
| start_ptt_session | pytest + scripted harness | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/integration/start_ptt_session_snapshot.json` | **PASS** |
| windows_system_aec_to_webrtc_apm | pytest + scripted harness | `windows_system_aec -> webrtc_apm` | `active` | `docs/audio_acceptance_evidence/integration/windows_system_aec_to_webrtc_apm_snapshot.json` | **PASS** |
| webrtc_apm_to_degraded_no_aec | pytest + scripted harness | `windows_system_aec -> webrtc_apm -> degraded_no_aec` | `degraded` | `docs/audio_acceptance_evidence/integration/webrtc_apm_to_degraded_no_aec_snapshot.json` | **PASS** |
| headset_clean_path | pytest + scripted harness | `headset_clean` | `active` | `docs/audio_acceptance_evidence/integration/headset_clean_path_snapshot.json` | **PASS** |
| transport_reconnect_without_backend_change | pytest + scripted harness | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/integration/transport_reconnect_without_backend_change_snapshot.json` | **PASS** |
| device_reset_and_xrun_escalation | pytest + scripted harness | `windows_system_aec -> webrtc_apm` | `active` | `docs/audio_acceptance_evidence/integration/device_reset_and_xrun_escalation_snapshot.json` | **PASS** |
| acceptance_builtin_windows_available | scripted harness + telemetry snapshot + session log | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/acceptance/acceptance_builtin_windows_available_snapshot.json` | **PASS** |
| acceptance_builtin_windows_fallback_webrtc | scripted harness + telemetry snapshot + session log | `windows_system_aec -> webrtc_apm` | `active` | `docs/audio_acceptance_evidence/acceptance/acceptance_builtin_windows_fallback_webrtc_snapshot.json` | **PASS** |
| acceptance_builtin_reference_pipeline_unhealthy | scripted harness + telemetry snapshot + session log | `windows_system_aec -> webrtc_apm -> degraded_no_aec` | `degraded` | `docs/audio_acceptance_evidence/acceptance/acceptance_builtin_reference_pipeline_unhealthy_snapshot.json` | **PASS** |
| acceptance_wired_headset_clean | scripted harness + telemetry snapshot + session log | `headset_clean` | `active` | `docs/audio_acceptance_evidence/acceptance/acceptance_wired_headset_clean_snapshot.json` | **PASS** |
| acceptance_reconnect_during_active_handsfree | scripted harness + telemetry snapshot + session log | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/acceptance/acceptance_reconnect_during_active_handsfree_snapshot.json` | **PASS** |
| soak_long_running_handsfree | scripted harness + fault injection + telemetry snapshot | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/soak/soak_long_running_handsfree_snapshot.json` | **PASS** |
| soak_repeated_reconnects | scripted harness + fault injection + telemetry snapshot | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/soak/soak_repeated_reconnects_snapshot.json` | **PASS** |
| soak_repeated_tts_and_barge_in | scripted harness + fault injection + telemetry snapshot | `windows_system_aec` | `active` | `docs/audio_acceptance_evidence/soak/soak_repeated_tts_and_barge_in_snapshot.json` | **PASS** |
| soak_backlog_playback_stagnation_detection | scripted harness + fault injection + telemetry snapshot | `windows_system_aec` | `recovering` | `docs/audio_acceptance_evidence/soak/soak_backlog_playback_stagnation_detection_snapshot.json` | **PASS** |
| soak_device_reset_xrun_fault_injection | scripted harness + fault injection + telemetry snapshot | `windows_system_aec -> webrtc_apm` | `active` | `docs/audio_acceptance_evidence/soak/soak_device_reset_xrun_fault_injection_snapshot.json` | **PASS** |

## Poznámky

- Každý scénář má vedle snapshotu i odpovídající `*.jsonl` session log a `*_verdict.json` soubor ve stejné složce.
- Acceptance scénáře jsou navržené jako deterministické scripted runs bez nutnosti reálného HW v CI.
- Hardwarově závislé vlastnosti notebookového a headset prostředí jsou oddělené od produkčního rozhodování a pokryté simulací/fault injection harnessy.
