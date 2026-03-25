# Gap Audit Audio Architektury

Datum: 2026-03-24

## Účel

Tento dokument porovnává:

- forenzní audit v [`audio (1).txt`](C:/Users/provo/Downloads/audio%20(1).txt)
- navrženou cílovou architekturu v [`audio_architektura_koncept (1).md`](C:/Users/provo/Downloads/audio_architektura_koncept%20(1).md)
- skutečnou implementaci v aktuálním stromu KajovoChat

Cíl není obhajovat aktuální stav, ale přesně oddělit:

- co je doručené,
- co je doručené jen částečně,
- co doručené není,
- a co je potřeba udělat, aby byl architektonický návrh skutečně naplněný.

## Shrnutí

Nová architektura není implementovaná beze zbytku.

Je doručená hlavně:

- orchestrace relace přes `AudioSessionManager`,
- explicitní backend policy a session state model,
- nový production naming backendů,
- řízený fallback chain,
- lepší session telemetrie a recovery logika.

Není ale doručené to nejzásadnější z auditu:

- samostatná nativní capture větev pro `windows_system_aec`,
- samostatný produkční `webrtc_apm` fallback backend,
- odstranění hybridního per-block DSP rozhodování z produkční notebookové cesty,
- samostatný headset/no-AEC produkční režim.

## Aktualizace po implementační vlně 2

Od původního gap auditu se posunuly tři důležité věci:

- session-owned audio ownership už není jen návrh; `AudioSessionManager` nově vytváří `DuplexAudioSession`, který vlastní render i capture I/O,
- produkční headset/no-AEC režim `headset_clean` je implementovaný jako samostatná backend policy větev,
- produkční režimy `windows_system_aec` a `webrtc_apm` už nepoužívají custom DSP výstup jako implicitní první volbu; custom větev zůstává laboratorní nebo nouzová.
- `windows_system_aec` už při přítomném APO driveru běží jako systémová capture větev bez app-level reference gate.
- `webrtc_apm` už neběží jako hybridní příměs custom DSP větve, ale jako samostatný produkční runner.

Další posun po implementační vlně 3:

- `ConversationWorker` už pro backlog, reference, capture queue a lipsync čte primárně přes session-owned `DuplexAudioSession`,
- `DuplexAudioSession` má vlastní runtime snapshot a soft-clock metriky,
- `AdaptiveEchoCanceller` už má oddělenou reference-prep vrstvu a backend-specific helpery pro native/WebRTC větev.

To zmenšuje rozdíl mezi návrhem a implementací, ale neuzavírá ho úplně. Největší nedoručená část zůstává plně systémová Windows capture integrace bez helper/fallback vrstvy.

## Doručeno

### 1. Session orchestrace byla oddělena od GUI workeru

Tohle je skutečně doručené.

V repu existuje:

- [`AudioSessionManager`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py)
- [`AecEngine`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py)
- [`DuplexDeviceGraph`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py)
- [`RecoverySupervisor`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py)
- [`RealtimeTransportBridge`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py)

`ConversationWorker` už část životního cyklu opravdu deleguje do session vrstvy a není to jen kosmetické přejmenování.

### 2. Backend policy je explicitní a auditovatelná

Tohle je také doručené.

Canonical režimy jsou dnes:

- `windows_system_aec`
- `webrtc_apm`
- `degraded_no_aec`
- `custom_lab`

To je vidět v:

- [`settings.py`](C:/GitHub/KajovoChat/kajovochat/settings.py)
- [`audio_session_manager.py`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py)
- [`audio_architecture.md`](C:/GitHub/KajovoChat/docs/audio_architecture.md)

Oproti starému stavu je to výrazně čistší.

### 3. Session-level fallback chain a recovery logika jsou doručené

Audit požadoval řízený fallback místo chaotického přepínání. To je implementované částečně dobře:

- session start probíhá přes backend selection chain
- fallback už není jen nahodilé přepínání uvnitř GUI smyčky
- existují explicitní `failure_reason`, `fallback_reason` a `degradation_cause`
- fallback už umí reagovat nejen na transport, ale i na některé audio health podmínky

### 4. Telemetrie a provozní logování jsou výrazně lepší

Tohle je doručené nad rámec původního stavu:

- session-level state logy
- backend selection logy
- reference health logy
- AEC summary a AEC diag
- tooling pro analýzu logů

To je reálný posun a není to jen dokumentační vrstva.

## Částečně doručeno

### 1. `DuplexDeviceGraph` existuje, ale není to skutečný duplex engine

V návrhu měl `DuplexDeviceGraph` držet capture i render pod jedním clockem a jedním session frame modelem.

V implementaci je to dnes jen držák na:

- `AudioPlayer`
- `RealtimeMicStream`

To je vidět v [`audio_session_manager.py`](C:/GitHub/KajovoChat/kajovochat/services/audio_session_manager.py), kde graph vlastní oddělené objekty, ale `_start_session()` dál otevírá samostatný `RealtimeMicStream` a player vzniká zvlášť v GUI vrstvě.

Výsledek:

- session orchestrace je lepší,
- ale audio I/O architektura zůstává rozdělená.

To není skutečné naplnění požadavku „jediný duplexní audio engine jako primární cesta“.

Aktualizace:

- ownership se už přesunul do session vrstvy přes `DuplexAudioSession`,
- `ConversationWorker` už není primární owner `AudioPlayer`,
- worker už navíc čte runtime stav primárně přes `DuplexAudioSession`,
- `DuplexAudioSession` už běží pod jedním `sounddevice.Stream` callbackem a sdílí společný render/capture clock.

### 2. `windows_system_aec` je implementované jako produkční systémová capture větev

Tohle je nyní z větší části doručené.

Při přítomném APO driveru už `windows_system_aec` neběží jako další app-level AEC procesor závislý na playback referenci. Worker tuto větev bere jako systémově zpracovanou capture cestu:

- reference není podmínkou pro běh `windows_system_aec`,
- blok se neposílá do hybridní custom/WebRTC arbitráže,
- backend se v diagnostice zapisuje jako systémová capture větev.

Pořád ale zůstává poslední mezera: systémová vazba je v uživatelském prostoru pořád zprostředkovaná helper/probe vrstvou a ne čistým WASAPI/WDK capture kontraktem bez mezivrstvy.

### 3. Produkční fallback na WebRTC je doručený jako samostatný běh

`webrtc_apm` už není jen další větev uvnitř hybridního custom rozhodování. Produkční běh se teď dispatchuje samostatně:

- používá vlastní produkční runner,
- neadaptuje custom NLMS větev jako mezikrok,
- při selhání vrací vlastní `webrtc_*` důvody místo propadu do custom výstupu.

Sdílená zůstává jen reference prep a základní alignment pomocná vrstva. To je přijatelný kompromis; produkční fallback už ale není hybridní custom cesta.

### 4. Recovery podle kvality backendu je jen částečně dotažené

Pozitivní změna je, že session manager už umí fallback i při dlouhodobě slabém `windows_system_aec`.

To je výrazně lepší než starší verze.

Pořád ale nejde o plně oddělený produkční DSP engine s minimem runtime větvení. Část rozhodování zůstává v per-block hot path.

## Nedoručeno

### 1. Jediný duplexní audio engine pod jedním frame clockem

Tohle je nyní doručené.

`DuplexAudioSession` už nevlastní oddělený `AudioPlayer` a `RealtimeMicStream` jako skutečné streamy. Uvnitř běží jeden `sounddevice.Stream`, který v jednom callbacku:

- odebírá render chunk z playback bufferu,
- aktualizuje playback level, lipsync a echo reference,
- zachytává mikrofonní chunk,
- timestampuje render i capture jedním monotonic clockem,
- plní jednotnou capture frontu `CapturedAudioChunk`.

`player` a `mic` zůstávají jen jako tenké kompatibilní pohledy nad stejnou duplex session.

### 2. Skutečný render reverse stream jako first-class vstup do AEC

Tohle doručené není.

Návrh výslovně požadoval, aby AEC dostával skutečnou render větev v přesném časovém rámci. Aktuální implementace pořád pracuje s referencí vznikající z `AudioPlayer` bufferingu a navazujících fallbacků.

To je lepší než původní stav, ale pořád je to aplikačně odhadovaný model, ne skutečný duplex reverse stream.

### 3. WebRTC APM jako hlavní trvalý fallback engine

Tohle je nyní doručené v rámci současné Python/native architektury.

### 4. Vyřazení custom Python AEC z notebook production path

Tohle je nyní také doručené.

`custom_lab` zůstává v repu pro laboratorní a regresní použití, ale produkční režimy `windows_system_aec`, `webrtc_apm`, `headset_clean` a `degraded_no_aec` už nevrací custom výstup jako implicitní notebookovou produkční cestu.

### 5. Samostatný headset/no-AEC produkční režim

Tohle už je doručené.

Audit i koncept výslovně doporučovaly separátní režim typu `no_aec_headset_mode` nebo `headset_clean`.

V kódu dnes existuje canonical produkční režim `headset_clean`, který:

- vzniká deterministicky z topologie zařízení přes `AecEngine`,
- má vlastní backend chain,
- nechce playback reference,
- nepředstírá AEC a běží jako čistý passthrough pro headsetovou topologii.

### 6. Jednoznačný stavový model z návrhu není plně převzatý

Session state model existuje, ale není totožný s návrhem.

Návrh chtěl stavy jako:

- `initializing`
- `calibrating`
- `ready`
- `assistant_rendering`
- `double_talk`
- `barge_in_transition`

Implementace má jinou, zjednodušenou množinu stavů:

- `idle`
- `starting`
- `probing`
- `active`
- `degraded`
- `recovering`
- `stopping`
- `failed`

To samo o sobě nemusí být chyba, ale znamená to, že návrh nebyl realizován beze zbytku.

### 7. Nativní vrstva neposkytuje celý požadovaný kontrakt

Koncept požadoval, aby nativní helper poskytoval:

- raw mic frame
- processed mic frame
- render reference frame
- VAD / residual / ERLE-like indikátory
- přesný session timestamp nebo frame counter
- signalizaci underrun/overrun/resetu

Současný bridge a helper tento kontrakt kompletně neexponují.

Tím pádem ani Python vrstva ještě nemohla být opravdu zjednodušená na lehký orchestrátor.

## Doporučení z auditu a jejich stav

### Doručeno

- zavést `AudioSessionManager`
- zavést explicitní backend chain
- zpřesnit session-level telemetry
- zpřehlednit naming backendů

### Částečně doručeno

- řízený fallback místo čistě ad-hoc přepínání
- session state model
- native Windows směr
- oddělení orchestrace od GUI workeru

### Nedoručeno

- plně systémová Windows capture integrace bez helper/probe mezivrstvy
- headset/no-AEC produkční režim
- zjednodušení guardu na čistou provozní pojistku místo sanace DSP neurčitosti

## Praktický závěr

Současná implementace je lepší než původní stav a správně se posunula směrem k robustnější architektuře.

Není ale pravda, že by byl nový architektonický návrh realizován beze zbytku.

Poctivější popis stavu je tento:

- session orchestrace je refaktorovaná,
- backend policy je čistší,
- telemetrie a recovery jsou lepší,
- ale systémová Windows větev je pořád částečně podepřená helper/probe vrstvou
- a hlavní architektonický zlom z auditu ještě nenastal.

## Doporučené další kroky

1. Dotáhnout `windows_system_aec` až na čistou systémovou capture integraci bez helper/probe mezivrstvy.
