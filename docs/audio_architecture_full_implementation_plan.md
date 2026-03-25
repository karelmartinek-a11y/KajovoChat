# Plán plnohodnotné implementace cílové audio architektury

Datum: 2026-03-24  
Repo: KajovoChat

## Cíl

Tento plán je určen k dosažení plného souladu s cílovou architekturou audio stacku bez obcházení, bez náhražek a bez helper-driven kompromisů vydávaných za finální řešení.

Cílový stav znamená:

- skutečný jednotný duplexní audio engine,
- skutečný systémový Windows capture AEC backend tam, kde je k dispozici,
- skutečný samostatný WebRTC APM duplex fallback backend,
- úplné vyřazení custom Python AEC z produkční notebookové cesty,
- `main.py` redukované na orchestrace UI a session eventů,
- explicitní kontrakty a stavový automat,
- acceptance důkazy proti požadovaným kritériím.

## Zásady realizace

1. Nesmí se vydávat přechodové řešení za finální architekturu.
2. Produkční cesty nesmí být založené na oportunistické hybridní arbitráži po framech.
3. Každá vlna musí zavřít konkrétní architektonický gap, ne jen zlepšit chování.
4. Kód musí být po každé vlně testovatelný a dokumentace musí odpovídat skutečnému stavu.
5. `custom_lab` zůstává laboratorní režim a nesmí znovu prorůst do produkční notebookové cesty.

## Pracovní proudy

### Proud A: Kontrakty a stavový model

Cíl:
- zavést explicitní typy `CaptureFrame`, `RenderFrame`, `SessionHealth`
- zavést plný stavový automat audio relace

Výstupy:
- `audio/contracts.py`
- `audio/session_state.py`
- testy přechodů stavového automatu

### Proud B: Session runtime a oddělení od GUI

Cíl:
- dostat veškerý produkční audio runtime z `main.py`
- nechat `main.py` jen jako UI orchestrace vrstvu

Výstupy:
- session runtime worker pod `audio/session_runtime.py`
- eventy `on_audio_state_changed`, `on_barge_in`, `on_session_health`, `on_backend_degraded`, `on_capture_ready`

### Proud C: Modulární rozpad audio vrstvy

Cíl:
- rozdělit dnešní monolit do cílových modulů

Výstupy:
- `audio/session_manager.py`
- `audio/device_graph.py`
- `audio/aec_backends/windows_system.py`
- `audio/aec_backends/webrtc_apm.py`
- `audio/voice_gate.py`
- `audio/telemetry.py`
- `audio/recovery.py`

### Proud D: Windows system AEC backend

Cíl:
- nahradit helper/probe orientovaný mezikrok skutečným session backendem

Výstupy:
- session-oriented Python bridge
- session-oriented native API
- health snapshot
- frame read/write kontrakt

### Proud E: WebRTC APM duplex backend

Cíl:
- vybudovat samostatný trvalý fallback engine

Výstupy:
- vlastní backend modul
- fixní 10ms duplex loop
- reverse stream vždy před process stream
- trvalá konfigurace AEC/NS/AGC/VAD

### Proud F: VoiceGate, Telemetry, Recovery

Cíl:
- udělat z těchto vrstev skutečné provozní služby, ne jen obálky

Výstupy:
- centralizovaná barge-in politika
- centralizovaný hold-off
- centralizovaný echo reject
- explicitní backend health model
- samostatný recovery supervisor

### Proud G: Acceptance a soak důkazy

Cíl:
- uzavřít architekturu důkazy, ne dojmem

Výstupy:
- integrační testy
- fallback/degraded testy
- barge-in testy
- soak scénáře
- jasný acceptance report

## Fázový plán

### Fáze 1: Kontrakty a stavový automat

Cíl:
- vytvořit tvrdý datový a stavový základ pro zbytek architektury

Konkrétní práce:
- vytvořit `CaptureFrame`, `RenderFrame`, `SessionHealth`
- vytvořit nový `SessionState` přesně podle návrhu
- vytvořit převodní vrstvu z dnešních dict payloadů na explicitní kontrakty
- doplnit testy stavů a přechodů

Definition of done:
- žádná nová produkční session logika už nepřenáší mezi vrstvami volné dicty tam, kde mají být kontrakty
- stavový model pokrývá celý navržený lifecycle

### Fáze 2: Vytažení audio runtime z main.py

Cíl:
- odstranit produkční audio runtime z GUI vrstvy

Konkrétní práce:
- přesunout realtime audio loop
- přesunout frame-level drop/keep logiku
- přesunout backend-aware audio rozhodování
- ponechat v `main.py` jen UI, signály a orchestrace

Definition of done:
- `main.py` už neobsahuje produkční frame-level audio runtime

### Fáze 3: Modulární split audio/*

Cíl:
- fyzicky rozdělit architekturu podle návrhu

Konkrétní práce:
- rozdělit `audio_service.py`
- rozdělit `audio_session_manager.py`
- přesunout logiku do cílových modulů

Definition of done:
- cílové moduly existují a jsou hlavním místem runtime logiky
- starý monolit už není produkční centrum architektury

### Fáze 4: Skutečný session backend pro windows_system_aec

Cíl:
- dostat `windows_system_aec` na session-oriented systémovou cestu

Konkrétní práce:
- nahradit frame-processor ABI session ABI
- zavést session open/start/read/write/health/stop/close kontrakt
- odstranit helper/probe logiku z role produkčního jádra

Definition of done:
- `windows_system_aec` už není jen bridge přes lokální frame processor

### Fáze 5: Skutečný WebRTC APM duplex backend

Cíl:
- udělat z `webrtc_apm` plně samostatný fallback engine

Konkrétní práce:
- vyvést WebRTC backend do vlastního modulu
- zavést vlastní session runtime a duplex processing
- odstranit závislost na hybridní cestě v `AdaptiveEchoCanceller`

Definition of done:
- `webrtc_apm` běží jako samostatný backend se svým kontraktem

### Fáze 6: VoiceGate, Telemetry, Recovery

Cíl:
- uzavřít provozní vrstvy podle návrhu

Konkrétní práce:
- centralizovat VAD a barge-in pravidla
- centralizovat hold-off a double-talk politiku
- převést telemetry na `SessionHealth`
- dovést recovery supervisor do plného rozsahu

Definition of done:
- guard už není rozptýlený po workeru a AEC runtime

### Fáze 7: Acceptance a soak

Cíl:
- prokázat architekturu v provozu

Konkrétní práce:
- integrační testy start/stop
- backend fallback testy
- degraded testy
- barge-in testy
- soak scénáře
- acceptance report

Definition of done:
- lze poctivě prohlásit, že návrh je naplněný nejen kódem, ale i důkazy

## Závislosti mezi fázemi

- Fáze 1 je podmínka pro všechny další
- Fáze 2 musí předcházet finálnímu acceptance uzavření
- Fáze 3 může běžet částečně souběžně s Fází 2, ale až po zavedení kontraktů
- Fáze 4 a 5 mohou běžet paralelně, pokud už existují kontrakty a session runtime hranice
- Fáze 6 musí dorovnat oba backendy před acceptance
- Fáze 7 je až poslední uzavírací vlna

## Orchestrace asistentů

### Asistent 1: Session runtime a `main.py`

Zodpovědnost:
- Fáze 1 a 2
- kontrakty, stavový model, vytažení runtime z GUI workeru

### Asistent 2: Windows system AEC backend

Zodpovědnost:
- Fáze 4
- Python bridge, native ABI, session backend kontrakt, testy a packaging dopady

### Asistent 3: WebRTC backend a provozní vrstvy

Zodpovědnost:
- Fáze 5 a 6
- samostatný `webrtc_apm` engine, `VoiceGate`, `AudioTelemetry`, `RecoverySupervisor`

### Hlavní orchestrátor

Zodpovědnost:
- drží architektonické invariants
- řeší konflikty mezi proudy
- slučuje rozhraní a dokumentaci
- provádí průběžné integrační testy

## Invarianty, které nesmí být porušeny

1. Produkční notebooková cesta nikdy nesmí znovu spadnout na `custom_lab`.
2. `main.py` nesmí po refaktoru znovu nabobtnat o frame-level audio runtime.
3. Každý backend musí mít explicitní a auditovatelný session kontrakt.
4. Každá vrstva musí mít jasnou odpovědnost.
5. Dokumentace musí po každé vlně odpovídat skutečnému kódu.

## Co se nesmí vydávat za hotové

- helper-backed Windows bridge jako finální systémový backend
- čistší dispatch v jednom monolitu jako finální modulární architektura
- dict payloady jako náhrada explicitních kontraktů
- několik dobrých ručních relací jako náhrada acceptance důkazů
- jednotkové testy jako náhrada integračního a soak ověření

## Finální kritérium dokončení

Plán je hotový teprve tehdy, když bude pravda všechno:

1. `windows_system_aec` je skutečný session-oriented systémový backend.
2. `webrtc_apm` je skutečný samostatný duplex fallback backend.
3. `main.py` je jen UI orchestrace vrstva.
4. `CaptureFrame`, `RenderFrame`, `SessionHealth` jsou explicitní produkční kontrakty.
5. Stavový automat audio relace odpovídá návrhu.
6. `VoiceGate`, `AudioTelemetry`, `RecoverySupervisor` jsou skutečné vrstvy.
7. Kód je fyzicky rozdělený do cílových audio modulů.
8. Acceptance kritéria jsou uzavřená důkazy.

