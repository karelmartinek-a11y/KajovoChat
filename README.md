KájovoChat (Windows Desktop, Python)

Co to dělá
- Fullscreen/maximalizované okno s titulkem „KájovoChat“
- Hands-free (výchozí): klik na Měsíc = režim „poslech → přepis (Whisper) → streamovaná odpověď → TTS → znovu poslech“
- Push-to-talk: držím Zeměkouli a mluvím, uvolnění ukončí záznam (jedno kolo)
- Streaming odpovědi v UI (text se zobrazuje postupně)
- Stabilní audio playback + barge-in: během „Thinking/Speaking“ můžeš začít mluvit a tím okamžitě přerušíš přehrávání a zrušíš běžící generování
- Deterministické auditní logy do JSONL (seq + ts_mono_ns) + čitelný TXT sidecar
- Nastavení: jazyk konverzace (Auto/cs/en/de/sk/fr), tykání/vykání, temperature, max output tokens, hlas a rychlost TTS, styl a délka odpovědi, adresář logů, výběr chat modelu
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
   - nastav hlas a rychlost TTS
   - případně „Načíst modely“ a vyber chat model
3) Klikni SAVE pro uložení jako výchozí

Logy
- Ukládají se do vybraného adresáře:
  - kajovochat_YYYYMMDD_HHMMSS.txt   (čitelné)
  - kajovochat_YYYYMMDD_HHMMSS.jsonl (strojově zpracovatelné; deterministické pořadí přes `seq`)

Poznámky
- Hands-free dělá krátkou kalibraci šumu při startu a dynamicky upraví práh VAD (viz `vad_calibration_s`, `vad_multiplier` v `kajovochat/settings.py`).
- Pokud mic nejde otevřít pro barge-in monitor současně s jinou aplikací, monitor se vypne a aplikace pokračuje bez něj (bez pádu).
- Pokud se TTS nepřehraje, zkontroluj výchozí zařízení Windows nebo nastav v settings.py `input_device`/`output_device` (indexy zařízení z knihovny sounddevice).
