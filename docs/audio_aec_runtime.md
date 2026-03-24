# Audio AEC Runtime

## Aktualni stav

Aktualni audio stack kombinuje:

- vlastni guard a telemetrii v `kajovochat/main.py`
- vlastni adaptivni AEC fallback v `kajovochat/services/audio_service.py`
- volitelny backend `aec-audio-processing` pro WebRTC AEC cisteni vhodnych echo-only bloku

Na realnem HW uz logy potvrzuji opakovane funkcni echo odeect. Nejde ale jeste o plne stabilni reseni napric celou relaci.

## Co se ted pouziva

- Timestampovane mic chunky
- Playback reference podle casu capture chunku
- Kratky cache fallback na posledni dobre reference okno
- Stabilizace runtime latence hysterzi
- `webrtc_success` signal pro bloky, kde WebRTC backend realne pomohl

## Co sledovat v session logu

Hlavni zdroj je session `.jsonl` log v adresari logu aplikace. Pro kazdou relaci jsou nejdulezitejsi zaznamy `aec_diag` a `aec_summary`.

Klicove metriky:

- `reference_miss_ratio`
- `reference_ready_ratio`
- `aligned_ratio`
- `strong_alignment_ratio`
- `avg_quality_when_aligned`
- `avg_delay_error`
- `backend`
- `ws`

## Prakticka interpretace

- Vysoke `reference_miss_ratio` znamena problem v reference pipeline.
- Nizke `aligned_ratio` znamena problem v coarse delay nebo alignment vrstve.
- `backend=webrtc` a `ws=on` znamena, ze WebRTC backend blok realne vycistil.
- Nizky `residual` a vysoke `improve` znamenaji funkcni echo odeect.
- Vysoke `avg_delay_error` znamena nestabilni runtime latenci.

## Soucasne zname limity

- V casti relace se porad objevuje `reference_miss`.
- Alignment neni stabilni na vsech blocich.
- Double-talk ochrana je funkcni v testech, ale na realnem HW zatim nebyla ve vetsim mnozstvi logu potvrzena.
- Reseni je vyrazne lepsi nez puvodni stav, ale stale nejde o plne systemovy AEC engine.

## Doporuceny dalsi postup

1. Sbirat dalsi realne session logy z bezneho pouzivani.
2. Doladovat reference pipeline podle `reference_miss_ratio`.
3. Doladovat guard fallback podle bloku s `backend=webrtc` a `ws=on`.
4. Dalsi velke DSP refaktory delat az pokud se reference pipeline stabilizuje a stale zustane nizke `avg_quality_when_aligned`.
