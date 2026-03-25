# Audio Architecture

## Cíl refaktoru

Audio vrstva je nově rozdělena tak, aby produkční relace nepřepínala mezi několika AEC strategiemi bez jasné politiky. Notebook s integrovaným mikrofonem a reproduktory má používat pevně definovanou produkční backendovou cestu:

1. `windows_system_aec`
2. `webrtc_apm`
3. `degraded_no_aec`

`custom_lab` zůstává pouze laboratorní režim pro DSP experimenty a regresní testy. Není produkční default.

## Hlavní vrstvy

### `AudioSessionManager`
Řídí lifecycle relace. Startuje hands-free/PTT session, provádí backend probing, zapisuje session telemetrii, přepíná AEC mód řízeným fallbackem a drží session model nad GUI workerem. `ConversationWorker` už pouze deleguje session orchestrace do této služby a zůstává hlavně UI/DSP integračním bodem.

### `DuplexDeviceGraph`
Drží duplexní audio topologii relace: vybraný input device, output device a session-owned `DuplexAudioSession`. Ten už běží pod jedním `sounddevice.Stream` callbackem a drží společný render/capture clock. `player` a `mic` v graphu zůstávají jen jako kompatibilní aliasy pro worker a GUI smyčku, ale nejsou to samostatné streamy.

### `AecEngine`
Nese explicitní backend policy. Vrací produkční backend chain podle zvoleného režimu a odděluje session-level rozhodnutí od DSP implementace v `AdaptiveEchoCanceller`.

### `VoiceGate`
Zapouzdřuje stav mikrofonní brány a přechod do transkripce. Session manager přes něj ovládá, kdy smí capture stream posílat data do transportu.

### `RealtimeTransportBridge`
Zapouzdřuje konfiguraci a lifecycle `RealtimeService`. `main.py` už nemusí přímo držet hlavní rozhodování o websocket session update, connect/reconnect a callback wiring.

### `AudioTelemetry`
Drží session-level metriky potřebné pro recovery a diagnostiku: requested/selected backend, fallback reason, device fingerprint, reference health, session timing a počítadla recovery pokusů.

### `RecoverySupervisor`
Provádí řízený reconnect a fallback orchestration. Obnovuje realtime transport podle session state a telemetrie místo ad-hoc flagů v GUI workeru. Současně drží anti-oscillation guard, aby se backendy nepřepínaly každých pár bloků.

## Session state model

Vnitřní audio relace používá explicitní session state machine:

- `idle`
- `starting`
- `probing`
- `active`
- `degraded`
- `recovering`
- `stopping`
- `failed`

UI dál používá jemnější prezentační stavy jako `listening`, `transcribing`, `thinking` a `speaking`, ale session manager rozhoduje o životním cyklu relace a o stabilitě audio pipeline.

## Produkční backend selection

Canonical AEC policy názvy jsou nyní:

- `windows_system_aec` – produkční default
- `webrtc_apm` – samostatný produkční fallback backend s vlastním během mimo custom DSP větev
- `headset_clean` – produkční režim pro headset topologii bez AEC
- `degraded_no_aec` – nouzový průchozí režim bez AEC
- `custom_lab` – laboratorní DSP režim

Výběr je deterministický a auditovatelný. V jedné produkční session je aktivní vždy jediná produkční AEC cesta. Pokud backend selže, session manager provede řízený přechod na další režim v chainu místo neustálého blokového přepínání.

## Integrace s GUI a realtime vrstvou

`ConversationWorker` nadále emituje Qt signály a drží guard/AEC diagnostiku, ale session entry points `start_handsfree`, `ptt_pressed`, `ptt_released`, reference-health reporting a reconnect lifecycle už delegují do `AudioSessionManager`.

Nově už ani nevlastní audio I/O jako primární owner. Render a capture jsou sjednocené pod `DuplexAudioSession`, který zakládá a ukončuje `AudioSessionManager`; worker si z něj jen čte kompatibilní aliasy a jednotný runtime snapshot.

To vytváří čistší hranici:

- GUI: tlačítka, signály, vizualizace, captions
- session services: lifecycle, transport, recovery, voice gate, backend policy
- audio DSP: capture/playback/AEC/reference processing

## Konfigurace

Konfigurace nyní rozlišuje produkční a diagnostické volby:

- `audio_aec_mode` – canonical backend request
- `audio_device_mode` – explicitní device class override (`auto`, `notebook_builtin`, headset, speakers…)
- `audio_session_profile` – `production` / `diagnostic` / `lab`
- `audio_diagnostics_enabled` – zapíná rozšířené logy bez míchání do produkčního fallback chainu

Bezpečný default je `windows_system_aec` + `audio_device_mode=auto` + `audio_session_profile=production`. Pokud je v systému nainstalovaný APO driver, `windows_system_aec` běží jako systémová capture větev bez app-level reference gate. Pokud systémová cesta není dostupná nebo se zhorší její zdraví, session manager přechází na samostatný `webrtc_apm` fallback. Při headsetové topologii policy přepne na `headset_clean` jako explicitní produkční no-AEC větev.
