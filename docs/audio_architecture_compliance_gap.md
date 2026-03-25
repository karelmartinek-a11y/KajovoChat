# Přesný compliance gap list proti cílové architektuře audio stacku

Datum: 2026-03-25  
Repo: KajovoChat

## Účel

Tento dokument je přísný checklist proti návrhu:

`Koncept a architektura audio stacku pro přirozenou komunikaci notebook ↔ hlasový chatbot`

Stavy:

- `SPLNĚNO`
- `ČÁSTEČNĚ`
- `NESPLNĚNO`

Pravidlo hodnocení:

- kompatibilitní wrapper se počítá jen tehdy, pokud už skutečně tvoří samostatnou architektonickou hranici
- laboratorní nebo přechodová implementace se nepočítá jako plné doručení produkční vrstvy
- za `SPLNĚNO` považujeme jen to, co je zároveň viditelné v kódu, kontraktech i testech

## Přísný verdikt

### Skutečně splněno

- `SPLNĚNO`: existuje skutečný jednotný duplexní audio engine pod jedním callbackem a jedním session clockem
- `SPLNĚNO`: backend chain je explicitní a deterministický
- `SPLNĚNO`: canonical režimy v nastavení odpovídají cílovému směru
- `SPLNĚNO`: custom Python AEC už není výchozí produkční notebooková cesta
- `SPLNĚNO`: explicitní datové kontrakty `CaptureFrame`, `RenderFrame`, `SessionHealth` existují
- `SPLNĚNO`: existuje explicitní session state machine se stavy z cílového návrhu
- `SPLNĚNO`: fyzický split do `kajovochat/audio/*` modulů už existuje
- `SPLNĚNO`: produkční `webrtc_apm` a `windows_system_aec` backendy jsou fyzicky vytažené do `kajovochat/audio/aec_backends/*`
- `SPLNĚNO`: reference hold-off, cached-reference stav a echo/barge-in čítače už nejsou volné fieldy workeru, ale session-owned runtime stav `VoiceGate`
- `SPLNĚNO`: bootstrap audio stacku už není ručně sestavovaný ve workeru, ale v [`C:\GitHub\KajovoChat\kajovochat\audio\bootstrap.py`](C:\GitHub\KajovoChat\kajovochat\audio\bootstrap.py)
- `SPLNĚNO`: reconnect/response timing runtime stav už není volný worker field, ale session-owned transport runtime ve [`C:\GitHub\KajovoChat\kajovochat\audio\session_callbacks.py`](C:\GitHub\KajovoChat\kajovochat\audio\session_callbacks.py)
- `SPLNĚNO`: audio-specifické formátování `aec_diag` logů už není ve workeru/observeru jako inline schema, ale ve [`C:\GitHub\KajovoChat\kajovochat\audio\log_formatters.py`](C:\GitHub\KajovoChat\kajovochat\audio\log_formatters.py)
- `SPLNĚNO`: `VoiceGate` už drží i TTS hold-off okna a lokální potvrzení barge-in streaku

### Ještě ne plně splněno

- `ČÁSTEČNĚ`: `main.py` je výrazně tenčí, ale stále drží část UI/audio bridge a worker orchestrace
- `ČÁSTEČNĚ`: `VoiceGate` je výrazně silnější, ale ještě není jediná finální vrstva pro veškerou hlasovou UX politiku
- `ČÁSTEČNĚ`: `RealtimeTransportBridge`, `AudioTelemetry` a `RecoverySupervisor` jsou samostatné vrstvy, ale ještě ne v úplně konečném rozsahu všech provozních scénářů z návrhu
- `ČÁSTEČNĚ`: `windows_system_aec` už má session-oriented API, ale pořád stojí na helper/DLL vrstvě, ne na čistém systémovém backendu bez této mezivrstvy
- `ČÁSTEČNĚ`: acceptance a soak důkazy jsou silnější než dřív, ale ještě nepokrývají všechny cílové notebookové scénáře z bodu 14

## Compliance matrix

### 1. Notebooková konfigurace je first-class produkční cesta

Stav: `SPLNĚNO`

Poznámka:
- notebookový režim je primární backend policy
- existuje explicitní volba mezi `windows_system_aec`, `webrtc_apm`, `degraded_no_aec` a `headset_clean`

### 2. Jeden duplexní audio engine jako primární cesta

Stav: `SPLNĚNO`

Důkaz:
- [`C:\GitHub\KajovoChat\kajovochat\services\audio_service.py`](C:\GitHub\KajovoChat\kajovochat\services\audio_service.py) obsahuje `DuplexAudioSession`
- capture i render běží pod jedním `sounddevice.Stream`

### 3. Jednoznačně definované fallbacky

Stav: `SPLNĚNO`

Důkaz:
- backend chain je explicitní
- `windows_system_aec -> webrtc_apm -> degraded_no_aec`

### 4. Jasně oddělené vrstvy capture / render / AEC / VAD / transport / telemetry / recovery

Stav: `ČÁSTEČNĚ`

Důvod:
- fyzické moduly už existují
- ale část UI/audio bridge a worker glue ještě zůstává v [`C:\GitHub\KajovoChat\kajovochat\main.py`](C:\GitHub\KajovoChat\kajovochat\main.py)

### 5. Minimum runtime větvení uvnitř každého bloku

Stav: `ČÁSTEČNĚ`

Důvod:
- produkční backendy jsou oddělené
- ale [`C:\GitHub\KajovoChat\kajovochat\services\audio_service.py`](C:\GitHub\KajovoChat\kajovochat\services\audio_service.py) je stále silný koordinátor s historickou kompatibilitou

### 6. Jedna pravda o delayi a frame clocku

Stav: `SPLNĚNO`

### 7. AudioSessionManager jako jediné místo volby backendu

Stav: `SPLNĚNO`

### 8. DuplexDeviceGraph jako samostatná vrstva

Stav: `SPLNĚNO`

### 9. AecEngine jako abstrakce nad třemi režimy

Stav: `SPLNĚNO`

### 10. VoiceGate jako jednoduchá a stabilní provozní logika

Stav: `ČÁSTEČNĚ`

Důvod:
- `VoiceGate` už centralizuje `GateDecision`, `evaluate_capture_gate(...)`, reference hold-off, session-owned gate runtime stav i lokální `barge_in_confirmed`
- drží i TTS start/tail hold-off okna
- ještě ale není jediným místem pro veškeré UX politiky kolem hlasu a část rozhodování stále žije v runtime smyčce

### 11. RealtimeTransportBridge jako samostatná vrstva

Stav: `SPLNĚNO`

### 12. AudioTelemetry jako samostatná vrstva

Stav: `ČÁSTEČNĚ`

Důvod:
- vrstva existuje a má explicitní kontrakty
- ještě chybí plnější důkazy pro dlouhé produkční notebookové běhy a některé provozní metriky z návrhu

### 13. RecoverySupervisor jako samostatná vrstva

Stav: `ČÁSTEČNĚ`

Důvod:
- reconnect a backend fallback orchestrace už je samostatná
- ještě nejsou dotažené všechny varianty `xrun` a device reset scénářů

### 14. Primární režim: Windows systémové AEC

Stav: `ČÁSTEČNĚ`

Důvod:
- `windows_system_aec` už používá session-oriented API přes [`C:\GitHub\KajovoChat\kajovochat\services\windows_native_aec.py`](C:\GitHub\KajovoChat\kajovochat\services\windows_native_aec.py)
- pořád ale stojí na helper-backed native vrstvě, ne na zcela čisté systémové session integraci bez DLL bridge

### 15. Nativní vrstva poskytuje session backend API

Stav: `SPLNĚNO`

### 16. Produkční fallback: WebRTC APM duplex engine

Stav: `SPLNĚNO`

### 17. Co odstranit z produkční fallback cesty

Stav: `ČÁSTEČNĚ`

Důvod:
- produkční fallback už není oportunistické míchání custom a webrtc jako dřív
- stále však žije nad kompatibilní koordinací ve velkém `audio_service.py`

### 18. Nouzový režim degraded_no_aec

Stav: `ČÁSTEČNĚ`

Důvod:
- režim existuje a je explicitní
- ještě není plně dotažený jako produktový UX režim se všemi konzervativními pravidly z návrhu

### 19. Explicitní stavový automat audio relace

Stav: `SPLNĚNO`

### 20. Explicitní datové kontrakty CaptureFrame / RenderFrame / SessionHealth

Stav: `SPLNĚNO`

### 21. main.py má být jen orchestrace UI + session eventů

Stav: `ČÁSTEČNĚ`

Důvod:
- velká část audio runtime už byla vytažená
- bootstrap audio stacku, reconnect/response runtime stav i `aec_diag` formatter už jsou mimo worker
- stále ale zůstává část UI/audio state bridge a worker orchestrace

### 22. Rozpad do cílových modulů audio/*

Stav: `SPLNĚNO`

### 23. settings.py má explicitní backend názvy

Stav: `SPLNĚNO`

### 24. VoiceGate pravidla jsou centralizovaná

Stav: `ČÁSTEČNĚ`

Důvod:
- hold-off/reference runtime, echo-drop side-effecty, diag throttle, lokální barge-in potvrzení i TTS hold-off okna jsou už navázané na `VoiceGate`
- stále ale chybí úplné sjednocení celé hlasové UX politiky do jedné finální vrstvy

### 25. Telemetrie je redukovaná na cílový backend health model

Stav: `ČÁSTEČNĚ`

### 26. Recovery a health-driven UX jsou hotové

Stav: `ČÁSTEČNĚ`

### 27. Acceptance kritéria jsou krytá důkazy

Stav: `ČÁSTEČNĚ`

Důvod:
- máme integrační a soak-like testy
- stále chybí plný uzavřený důkaz pro všechny notebookové scénáře z bodu 14 návrhu

## Přesný zbytkový gap list

Pro skutečné tvrzení o 100% souladu ještě zbývá:

1. Dotáhnout `main.py` na čistou UI/session orchestrace vrstvu bez zbylého audio bridge glue.
2. Dovést `VoiceGate` do finální jediné provozní vrstvy pro kompletní hlasovou UX politiku.
3. Dovést `AudioTelemetry` a `RecoverySupervisor` na plný provozní rozsah včetně `xrun` a device reset scénářů.
4. Dodat acceptance a soak důkazy pro plné notebookové scénáře podle bodu 14 cílového návrhu.
5. Pokud trvá požadavek na zcela čistý OS backend bez mezivrstvy, nahradit helper-backed `windows_system_aec` čistou systémovou session integrací.

## Aktuální poctivý závěr

Architektura je dnes výrazně blíž cílovému návrhu než původní hybridní stav a velká část bodů už je opravdu doručená. Přesto ještě nelze poctivě tvrdit 100% soulad bez zbytku. Největší zbývající rozdíl už není v základní architektuře, ale v poslední míli:

- dotažení UI/session hranice,
- dotažení provozních vrstev,
- úplné acceptance důkazy,
- a případně odstranění helper-backed mezivrstvy u `windows_system_aec`.
