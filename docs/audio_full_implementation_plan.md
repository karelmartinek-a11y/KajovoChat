# Implementační plán k plnohodnotné audio architektuře bez obcházení a bez náhražek

Datum: 2026-03-24  
Repo: KajovoChat

## Účel

Tento dokument převádí přísný gap audit do skutečného implementačního programu.

Nejde o „jak to nějak splnit“.  
Jde o to, jak doručit **plnohodnotnou cílovou architekturu**, aby šlo poctivě říct:

- máme skutečný duplexní audio engine,
- máme skutečný systémový AEC tam, kde je k dispozici,
- jinak máme skutečný WebRTC APM duplex fallback engine,
- custom Python AEC není produkční notebooková cesta,
- `main.py` už není audio runtime,
- vrstvy a kontrakty odpovídají návrhu,
- acceptance kritéria jsou doložená.

## Řídicí zásady

1. Žádné kompatibilitní zkratky nesmí být vydávané za finální architekturu.
2. Produkční backendy jsou pouze:
   - `windows_system_aec`
   - `webrtc_apm`
   - `headset_clean`
   - `degraded_no_aec`
3. `custom_lab` zůstává pouze laboratorní a regresní režim.
4. `main.py` nesmí rozhodovat o audio runtime po framech.
5. Datové kontrakty musí být explicitní a stabilní.
6. Každá fáze má vlastní definition-of-done a nesmí se uzavřít jen „zelenými unit testy“.

## Kritické rozhodnutí

Plán má dvě větve:

- **větev A: architektonický refaktor Python runtime**
- **větev B: skutečný session-oriented native backend**

Bez uzavření obou větví nelze tvrdit 100% soulad.

## Cílový stav po dokončení

### Cílové moduly

Musí vzniknout tyto produkční moduly:

- `kajovochat/audio/contracts.py`
- `kajovochat/audio/device_graph.py`
- `kajovochat/audio/session_manager.py`
- `kajovochat/audio/aec_backends/windows_system.py`
- `kajovochat/audio/aec_backends/webrtc_apm.py`
- `kajovochat/audio/voice_gate.py`
- `kajovochat/audio/telemetry.py`
- `kajovochat/audio/recovery.py`
- `kajovochat/audio/transport_bridge.py`
- `kajovochat/audio/lab/custom_aec.py`

### Cílové kontrakty

Musí vzniknout explicitní typy:

- `CaptureFrame`
- `RenderFrame`
- `SessionHealth`
- `BackendHealthSnapshot`
- `BargeInDecision`
- `AudioSessionEvent`

### Cílové backendy

- `WindowsSystemAecBackend`
- `WebRtcApmBackend`
- `HeadsetCleanBackend`
- `DegradedNoAecBackend`

## Pracovní program

## Fáze 0: Zamrazení současného hybridu

### Cíl

Zabránit tomu, aby se během přestavby dál rozrůstal historický hybrid.

### Kroky

1. Označit `AdaptiveEchoCanceller` jako laboratorní/historický runtime.
2. Zastavit další rozšiřování guard heuristik v `main.py`.
3. Přestat přidávat nové produkční větve do `audio_service.py`.

### Definition-of-done

- žádná nová produkční logika už nepřibývá do `main.py`
- žádná nová backend volba už nevzniká v hybridním AEC bloku

## Fáze 1: Datové kontrakty a stavový model

### Cíl

Položit pevné rozhraní, bez kterého se zbytek přepisu znovu rozpadne.

### Implementace

1. Vytvořit [`kajovochat/audio/contracts.py`](C:/GitHub/KajovoChat/kajovochat/audio/contracts.py)
2. Zavést dataclasses:
   - `CaptureFrame`
   - `RenderFrame`
   - `SessionHealth`
   - `BackendHealthSnapshot`
   - `AudioSessionEvent`
3. Zavést nový `SessionState` přesně podle návrhu:
   - `idle`
   - `initializing`
   - `calibrating`
   - `ready`
   - `assistant_rendering`
   - `double_talk`
   - `barge_in_transition`
   - `recovering`
   - `degraded`
   - `stopping`
   - `failed`
4. Zavést přechodovou tabulku a explicitní validační funkci přechodů.

### Co se musí přepsat

- [`C:\GitHub\KajovoChat\kajovochat\services\audio_session_manager.py`](C:\GitHub\KajovoChat\kajovochat\services\audio_session_manager.py)
- [`C:\GitHub\KajovoChat\kajovochat\main.py`](C:\GitHub\KajovoChat\kajovochat\main.py)
- všechny testy, které dnes pracují s volnými dict payloady

### Definition-of-done

- mezi vrstvami se nepřenášejí volné dict payloady tam, kde mají být kontrakty
- session state už neběží na improvizovaných string stavech
- existují jednotkové testy na přechody stavového automatu

### Paralelizace

Tuto fázi lze dělat paralelně ve dvou proudech:
- A1: datové kontrakty
- A2: stavový automat

## Fáze 2: Vytažení audio runtime z main.py

### Cíl

Přestat používat `main.py` jako produkční audio runtime.

### Implementace

1. Přesunout realtime audio loop z `main.py` do session vrstvy.
2. Přesunout frame-level drop/keep logiku do `VoiceGate`.
3. Přesunout backend-aware AEC runtime rozhodování mimo GUI worker.
4. Omezit `main.py` na:
   - UI orchestrace
   - stavové eventy
   - uživatelské akce
   - zobrazení health a chyb

### Co se musí přepsat

- [`C:\GitHub\KajovoChat\kajovochat\main.py`](C:\GitHub\KajovoChat\kajovochat\main.py)
- navazující testy workeru

### Definition-of-done

- `main.py` už neobsahuje frame-level audio runtime logiku
- `main.py` nesestavuje AEC rozhodnutí po blocích
- `main.py` pracuje jen s `AudioSessionEvent` a `SessionHealth`

### Paralelizace

Lze dělat souběžně s Fází 3, ale nesmí se mergeovat dřív než po Fázi 1.

## Fáze 3: Modulární rozpad audio stacku

### Cíl

Nahradit stávající monolit skutečnými vrstvami.

### Implementace

1. Vyvést `DuplexAudioSession` do `audio/device_graph.py`
2. Vyvést `AudioSessionManager` do `audio/session_manager.py`
3. Vyvést `VoiceGate` do `audio/voice_gate.py`
4. Vyvést `RealtimeTransportBridge` do `audio/transport_bridge.py`
5. Vyvést `RecoverySupervisor` do `audio/recovery.py`
6. Vyvést telemetrii do `audio/telemetry.py`
7. Přesunout laboratorní custom AEC do `audio/lab/custom_aec.py`

### Definition-of-done

- `audio_service.py` už není hlavní produkční monolit
- produkční audio runtime je čitelně rozdělený podle vrstev z návrhu
- staré kompatibilitní aliasy jsou buď odstraněné, nebo jasně omezené na přechodnou vrstvu

### Paralelizace

Tuto fázi lze rozdělit na:
- B1: device graph + contracts napojení
- B2: session manager + recovery
- B3: telemetry + voice gate + transport

## Fáze 4: Skutečný Windows system AEC session backend

### Cíl

Přestat se opírat o helper/probe mezivrstvu jako o „hotový systémový backend“.

### Implementace

1. Definovat skutečný Python-side session kontrakt:
   - `open_session(config)`
   - `start()`
   - `read_capture_frame(timeout_ms) -> CaptureFrame`
   - `write_render_frame(frame: RenderFrame)`
   - `get_health_snapshot() -> BackendHealthSnapshot`
   - `stop()`
   - `close()`
2. Přepsat [`C:\GitHub\KajovoChat\kajovochat\services\windows_native_aec.py`](C:\GitHub\KajovoChat\kajovochat\services\windows_native_aec.py) z frame processoru na session bridge.
3. Přepsat native helper/apo projekty tak, aby poskytovaly session-oriented kontrakt, ne jen `process(...)`.
4. Oddělit:
   - runtime session backend
   - packaging
   - install/probe diagnostiku
5. Probe a packaging už nesmí být hlavní důkaz, že backend je „produkčně hotový“.

### Native část

Je potřeba rozšířit:
- [`C:\GitHub\KajovoChat\native\windows_apo_helper`](C:\GitHub\KajovoChat\native\windows_apo_helper)
- případně i [`C:\GitHub\KajovoChat\native\windows_aec_helper`](C:\GitHub\KajovoChat\native\windows_aec_helper)

Na skutečné exporty typu:
- `kajovochat_apo_session_open`
- `kajovochat_apo_session_start`
- `kajovochat_apo_session_read_capture`
- `kajovochat_apo_session_write_render`
- `kajovochat_apo_session_get_health`
- `kajovochat_apo_session_stop`
- `kajovochat_apo_session_close`

### Definition-of-done

- `windows_system_aec` už neběží přes frame-level helper bridge
- Python už nevolá „zpracuj tento blok“, ale skutečně obsluhuje session backend
- systémový backend vrací `CaptureFrame` a `BackendHealthSnapshot`

### Paralelizace

Tato fáze se rozpadá na:
- C1: Python session bridge
- C2: native session ABI
- C3: packaging/install/runtime verification

## Fáze 5: Skutečný WebRTC APM duplex backend

### Cíl

Mít jeden skutečný fallback engine, ne jen čistší větev ve starém souboru.

### Implementace

1. Vytvořit [`kajovochat/audio/aec_backends/webrtc_apm.py`](C:/GitHub/KajovoChat/kajovochat/audio/aec_backends/webrtc_apm.py)
2. Zavést samostatnou session-oriented backend třídu:
   - `configure()`
   - `start()`
   - `push_render_frame(RenderFrame)`
   - `process_capture_frame(CaptureFrame) -> CaptureFrame`
   - `get_health_snapshot()`
   - `stop()`
3. Zajistit:
   - fixní 10 ms frame cadence
   - reverse stream vždy před process stream
   - jednotný `stream_delay_ms`
   - trvalou konfiguraci AEC/NS/AGC/VAD
4. Úplně odpojit produkční `webrtc_apm` od hybridního `AdaptiveEchoCanceller`.

### Definition-of-done

- `webrtc_apm` už není produkční větev uvnitř starého hybridního AEC bloku
- session manager pracuje s WebRTC backendem jako s plnohodnotným backendem

### Paralelizace

Lze dělat souběžně s Fází 4, ale merge až po Fázích 1 až 3.

## Fáze 6: VoiceGate, Telemetry, Recovery jako skutečné vrstvy

### Cíl

Nahradit rozptýlené heuristiky finálními vrstvami podle návrhu.

### VoiceGate

Musí centralizovat:
- VAD
- echo reject
- hold-off po TTS start/stop
- double-talk politiku
- barge-in politiku

### AudioTelemetry

Musí centralizovat:
- session-level health
- frame-level diag jen při problému
- agregované health score
- jednotný export

### RecoverySupervisor

Musí centralizovat:
- device reset
- stream xrun recovery
- restart session při selhání backendu
- degradaci bez pádu GUI

### Definition-of-done

- pravidla už nejsou rozptýlená mezi `main.py`, session manager a AEC vrstvou
- health-driven fallback rozhodování je centralizované

## Fáze 7: Skutečný degraded režim a headset režim

### Cíl

Dotáhnout provozní režimy, ne je jen formálně pojmenovat.

### Implementace

1. `headset_clean`:
   - explicitní no-AEC produkční větev
   - jiná barge-in politika
   - minimální echo logika
2. `degraded_no_aec`:
   - push-to-talk nebo polo-duplex
   - jasná signalizace v UI
   - konzervativní hold-off
   - žádný falešný hands-free příslib

### Definition-of-done

- oba režimy jsou skutečně odlišné produktové provozní režimy, ne jen jiné stringy backendu

## Fáze 8: Acceptance a soak důkazy

### Cíl

Uzavřít architekturu důkazy, ne dojmem.

### Minimální sada důkazů

1. 10 po sobě jdoucích hovorů na notebooku bez crash/deadlock.
2. Barge-in bez výrazného useknutí začátku věty.
3. Fallback z `windows_system_aec` do `webrtc_apm` bez pádu GUI.
4. Přechod do `degraded_no_aec` bez pádu GUI.
5. Logy jednoznačně ukážou backend, důvod fallbacku a stav session.

### Testy

Je potřeba doplnit:
- integrační testy session start/stop
- backend fallback testy
- degraded testy
- VoiceGate testy
- soak runner pro notebook profil

### Definition-of-done

- acceptance kritéria z návrhu jsou doložená artefakty, ne jen ručně popsaná

## Paralelní organizace práce

### Proud A: kontrakty + stavový automat

Může běžet hned.

### Proud B: vytažení runtime z main.py + modulární split

Může začít po návrhu kontraktů, ale merge až po stabilizaci Fáze 1.

### Proud C: native Windows session backend

Může běžet paralelně s Proud B, pokud je stabilní kontrakt z Fáze 1.

### Proud D: WebRTC session backend

Může běžet paralelně s Proud C.

### Proud E: VoiceGate/Telemetry/Recovery

Může běžet po Fázích 1 až 3 a současně se stabilizací backendů.

### Proud F: acceptance a soak

Musí běžet až nad sjednocenou architekturou, ne nad mezistavy.

## Definice hotovo

Za hotové to lze označit teprve tehdy, když platí současně:

1. `main.py` je jen UI orchestrace.
2. Produkční audio runtime je rozdělený do cílových modulů.
3. `windows_system_aec` je session backend bez helper/probe obezličky.
4. `webrtc_apm` je samostatný duplex backend.
5. `VoiceGate`, `AudioTelemetry`, `RecoverySupervisor` jsou skutečné vrstvy.
6. `custom_lab` je mimo produkční notebookovou cestu i architektonicky.
7. Datové kontrakty a stavový automat odpovídají návrhu.
8. Acceptance kritéria jsou doložená integračně.

## Praktický příkaz pro realizaci

Pokud se má tento plán opravdu odpracovat bez náhražek, je správné pořadí:

1. nejdřív kontrakty a stavový automat
2. pak vytažení runtime z `main.py`
3. pak fyzický rozpad modulů
4. pak session-native Windows backend a samostatný WebRTC backend
5. pak až VoiceGate/Telemetry/Recovery finální dotažení
6. nakonec acceptance a soak důkazy

Jakýkoliv jiný postup povede k tomu, že se znovu vrátí obezličky.

