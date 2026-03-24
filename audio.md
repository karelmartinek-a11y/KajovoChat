# Forenzni audit audio cesty

Datum auditu: 2026-03-23

## Rozsah

Audituje se cely tok zpracovani zvuku v notebooku:

- mikrofonni vstup
- vytvareni playback reference
- odhad latence
- custom AEC fallback
- WebRTC AEC pres `aec-audio-processing`
- guard a rozhodovani o dropu
- shutdown a reconnect chovani

Zamereni je na stav po poslednich upravach a na realne logy z provozu.

## Strucne reseni

Aktualni implementace uz neni jednovetvovy echo suppressor. Je to kombinace:

- timestampovane playback reference v `AudioPlayer`
- casove vazane cteni reference pro konkretnich zachycenych mic chunku
- vlastni adaptivni AEC fallback v `AdaptiveEchoCanceller`
- volitelny WebRTC backend pres `aec-audio-processing`
- guard vrstva, ktera kombinuje `similarity`, `residual`, `voice_likelihood`, `double_talk` a `reference_miss`

Prakticky vysledek z poslednich logu je nasledujici:

- WebRTC backend skutecne odecte echo na realnem HW
- reference pipeline je vetsinu casu dostupna
- custom fallback je stale slabsi cast systemu
- posledni cast opravn se zamerila na to, aby `custom` neprodukoval falezne pozitivni bloky a aby `webrtc` nebral prilis spatne kandidaty

## Forenzni audit toku

### 1. Mikrofonni vstup

Mic data jdou pres `RealtimeMicStream`, ktery dnes posila `CapturedAudioChunk` s PCM a timestampem zachyceni.

Relevantni kod:

- [`kajovochat/services/audio_service.py`](./kajovochat/services/audio_service.py#L1691)
- [`kajovochat/main.py`](./kajovochat/main.py#L1177)

Pozorovani:

- zachyceni je casove vazane, ne jen “posledni chunk”
- to zlepsilo parovani playback reference s tim, co mic realne slysel

### 2. Playback reference

`AudioPlayer` drzi ring buffer prehranych vzorku a vraci referenci pro konkretni capture cas.

Relevantni kod:

- [`kajovochat/services/audio_service.py`](./kajovochat/services/audio_service.py#L1388)
- [`kajovochat/services/audio_service.py`](./kajovochat/services/audio_service.py#L1572)

Pozorovani:

- reference uz nevznika jako nahodny tail
- je mozne vratit posledni odehrany usek pro dany mic chunk
- existuje i cache fallback na posledni dobrou referenci

Forenzni vyznam:

- puvodni problem nebyl jen v AEC algoritmu
- cast chyby byla v tom, ze AEC casto dostavala spatne nebo prazdne referencni okno

### 3. AEC jadro

`AdaptiveEchoCanceller` kombinuje:

- exhaustive alignment
- NLMS a ridge fallback
- WebRTC backend
- detekci double-talk

Relevantni kod:

- [`kajovochat/services/audio_service.py`](./kajovochat/services/audio_service.py#L551)
- [`kajovochat/services/audio_service.py`](./kajovochat/services/audio_service.py#L742)
- [`kajovochat/services/audio_service.py`](./kajovochat/services/audio_service.py#L924)

Pozorovani:

- WebRTC backend je dnes funkcni a na realnych blocich vraci `webrtc_success`
- v poslednich dobre fungujicich logu byl `webrtc_far` na nule nebo velmi nizky
- `custom` fallback byl postupne zprahovan, protoze casto daval nulovy prinos a matouci signal pro guard

### 4. Guard vrstva

Guard dnes rozhoduje nad:

- `similarity`
- `residual_level`
- `aec_quality`
- `voice_likelihood`
- `double_talk`
- `reference_miss`

Relevantni kod:

- [`kajovochat/main.py`](./kajovochat/main.py#L149)
- [`kajovochat/main.py`](./kajovochat/main.py#L1353)

Pozorovani:

- `effective_aec_quality` je zvysena pro `webrtc_success`
- `reference_miss` je stale dulezity signal pro drop logiku
- posledni verze uz nedovoli slabym `custom` blokum predstirat uspech

### 5. Shutdown a race condition

Byla nalezena a opravena race condition:

- thread mic loopu volal `append_audio_pcm16`
- mezitim uz byla `self._rt` nastavena na `None`
- vznikl crash `AttributeError: 'NoneType' object has no attribute 'append_audio_pcm16'`

Relevantni kod:

- [`kajovochat/main.py`](./kajovochat/main.py#L1457)
- [`kajovochat/main.py`](./kajovochat/main.py#L1498)
- [`kajovochat/main.py`](./kajovochat/main.py#L1529)

Stav po oprave:

- mic loop si bere lokalni referenci na `rt`
- shutdown zastavi loop driv, nez se realtime objekt zneplatni
- posledni app log uz ten crash neopakuje

## Binarni audit

### Co je v repozitari

Repo samotne neobsahuje vlastni native binarky ani vykonatelne soubory. Jde o Python aplikaci a cely audity prostor je ve zdrojacich.

### Runtime binarni zalezitosti

V `requirements.txt` jsou nejdolezitejsi nativni / binarni zavislosti:

- `PySide6`
- `moderngl`
- `numpy`
- `scipy`
- `aec-audio-processing`
- `sounddevice`
- `soundfile`
- `websocket-client`

Relevantni soubor:

- [`requirements.txt`](./requirements.txt)

Pozorovani:

- `aec-audio-processing` je jediny explicitni AEC binary backend
- pri testech se objevily SWIG deprecation warningy, ale ne runtime crash z binarky
- z hlediska forenzniho auditu to znamena, ze hlavni binarni riziko je dodavatelsky retezec a ABI kompatibilita, ne vlastni projektovy binary artifact

### Forenzni poznamka k bezpecnosti

V poslednich testech a logu neni zretelny indikat malicious binary behavior. Rizikem zustava:

- cizi wheel `aec-audio-processing`
- audio stack pres systemove device backendy
- ruzne native dependency ABI kombinace na Windows

## Log evidence

Nejdulezitejsi provozni logy:

- [`C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_145807.jsonl`](C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_145807.jsonl)
- [`C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_152617.jsonl`](C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_152617.jsonl)
- [`C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_161121.jsonl`](C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_161121.jsonl)
- [`C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_162002.jsonl`](C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_162002.jsonl)
- [`C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_170637.jsonl`](C:\Users\provo\Documents\ChatbotKajaLogs\kajovochat_20260323_170637.jsonl)

### Co logy ukazuji

- `reference_miss_ratio` se uz drzi nizko a v lepsich behach kleslo az k nekolika procentum
- `webrtc_success` je na reálnych blocich opakovane potvrzen
- problematicke `webrtc ws=off` bloky byly v posledni fazi odstranovany
- `custom_nonmiss_zero` se zmensilo, ale `custom` cast stale generuje nejslabsi bloky
- `avg_delay_error` se zlepsuje a pak zase kolisa podle toho, jak moc se povoli `custom` / `webrtc`

## Zaver

Forenzni zavr:

1. Puvodni hlavni chyba nebyla jen v algoritmu AEC, ale i v reference toku a shutdown race condition.
2. Race condition s `append_audio_pcm16` byla opravena.
3. WebRTC backend je realne funkcni a na dobrech blocich odecte echo.
4. Nejvetsi zbyvajici slabina je stale kompromis mezi `custom` fallbackem a mire povoleni `webrtc`.
5. Celkove je stav po poslednich upravach stabilnejsi, bez crashu a s mensim poctem spatnych deti `webrtc`, ale stale to neni plne systemove AEC reseni.

## Doporuceni

1. Drzet se zpetneho zemedeni `custom` fallbacku a nepridavat dalsi agresivni pravidla bez noveho logu.
2. Pokud bude chtit dalsi zlepseni, dalsi kroky delat uz jen podle novych session logu.
3. Pred vetsi zmenou znovu overit:
   - `reference_miss_ratio`
   - `webrtc_success`
   - `webrtc_far`
   - `custom_nonmiss_zero`
   - `avg_delay_error`

