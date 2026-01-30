KájovoChat (Windows Desktop, Python)

Co to dělá
- Fullscreen/maximalizované okno s titulkem „KájovoChat“
- Klik na Měsíc = nonstop hlasový režim (poslech → přepis → odpověď → hlas → znovu poslech)
- Tlačítko Zeměkoule = push‑to‑talk (jedno kolo: poslech → odpověď → hlas)
- Titulky přepisu nad Orbem + průběžný záznam do souboru (txt + jsonl) téměř v reálném čase
- Nastavení stylu/délky odpovědi, jazyka a pohlaví hlasu, adresáře logů, výběru modelu
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
   - nastav styl/délku/jazyk/pohlaví
   - vyber adresář logů
   - případně „Načíst modely“ a vyber chat model
3) Klikni SAVE pro uložení jako výchozí

Logy
- Ukládají se do vybraného adresáře:
  - kajovochat_YYYYMMDD_HHMMSS.txt  (čitelné)
  - kajovochat_YYYYMMDD_HHMMSS.jsonl (strojově zpracovatelné)

Poznámky
- VAD (detekce ticha) je jednoduchá (RMS). Pokud se nahrávání ukončuje moc brzy/pozdě, uprav v kajovochat/settings.py:
  vad_rms_threshold, vad_silence_ms.
- Pokud se TTS nepřehraje, zkontroluj výchozí zařízení Windows nebo nastav v settings.py input_device/output_device (indexy zařízení z knihovny sounddevice).
