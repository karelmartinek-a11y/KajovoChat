KájovoChat (Windows Desktop, Python)

Co to dělá
- Fullscreen/maximalizované okno s titulkem „KájovoChat“
- Realtime speech-to-speech přes OpenAI Realtime API (jedno stavové spojení, průběžné audio in/out)
- Hands-free: klik na Měsíc = kontinuální streaming mikrofonu; server-side VAD rozhoduje o koncích tahů a model odpovídá automaticky
- Push-to-talk: držím Zeměkouli a mluvím; uvolnění = commit audio + response.create
- Streaming odpovědi v UI (text se zobrazuje postupně)
- Streaming audio playback (odpověď se přehrává průběžně)
- Nastavení: jazyk konverzace (Auto/cs/en/de/sk/fr), tykání/vykání, hlas (voice), styl/délka odpovědi, adresář logů, výběr realtime modelu
- OPEN AI dialog pro vložení/uložení/smazání API key

Požadavky
- Windows 10/11
- Python 3.10+ (doporučeno 3.11)
- Funkční mikrofon a výstup zvuku

Instalace (1 krok)
1) V kořeni projektu spusť:
   python -m pip install -r requirements.txt

Spuštění (1 krok)
1) Dvojklik na run_kajovochat.bat
   nebo v terminálu:
   python -m kajovochat

První nastavení
1) Klikni OPEN AI a vlož API key → Uložit → Zavřít
2) Klikni NASTAVENÍ:
   - vyber jazyk konverzace (Auto nebo konkrétní)
   - nastav tykání/vykání
   - nastav hlas (voice) a realtime model (doporučeno: gpt-realtime)
3) Klikni SAVE pro uložení jako výchozí

Logy
- Ukládají se do vybraného adresáře:
  - kajovochat_YYYYMMDD_HHMMSS.txt   (čitelné)
  - kajovochat_YYYYMMDD_HHMMSS.jsonl (strojově zpracovatelné; deterministické pořadí přes `seq`)

Poznámky
- Realtime režim očekává PCM16 mono @ 24kHz. Pokud dané audio zařízení nepodporuje 24kHz, appka se pokusí otevřít zařízení na jeho defaultní vzorkovací frekvenci (často 48kHz) a provede resampling na 24kHz (mic) / z 24kHz (playback).
- „TTS“ v nastavení nyní znamená voice pro realtime (není to samostatné TTS volání).
