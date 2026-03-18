# Chatbot Kája

Desktop aplikace v `PySide6` pro hlasovou konverzaci přes OpenAI Realtime API.

## Co aplikace umí

- hands-free hlasovou konverzaci přes klik na hlavu asistenta
- push-to-talk přes tlačítko zeměkoule
- photo-based talking head widget s lipsyncem podle skutečně přehrávaného audia
- barge-in během mluvení asistenta
- softwarové tlumení self-hearing na notebookových reproduktorech

## Produktové nastavení

V běžném dialogu `Nastavení` zůstávají jen tyto volby:

- režim jazyka odpovědi
  - odpovídat jazykem uživatele
  - vždy odpovídat zvoleným jazykem
- pevný jazyk odpovědi
- styl odpovědi
  - stručný
  - vědecký s analýzou
  - normální

Model, hlas, VAD, rychlost výstupu a výběr audio zařízení jsou řízené interně.

## Požadavky

- Python `3.11+`
- funkční mikrofon a reproduktory nebo sluchátka
- platný OpenAI API klíč

## Instalace

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Spuštění

Preferovaný vstupní bod:

```bash
python -m kajovochat
```

Alternativní vstupní bod:

```bash
python app_gui.py
```

## OpenAI klíč

Po spuštění otevřete dialog `OpenAI` a vložte API klíč. Klíč se ve Windows ukládá přes DPAPI, na ostatních platformách přes systémový keyring, pokud je dostupný.

## Konfigurace

Konfigurace se ukládá do `settings.json` v uživatelském profilu aplikace. Ve Windows typicky:

```text
C:\Users\<uživatel>\AppData\Local\Kajovo\ChatbotKaja\settings.json
```

Technické logy se ukládají do:

```text
C:\Users\<uživatel>\Documents\ChatbotKajaLogs
```

## Struktura projektu

- `kajovochat/` hlavní aplikační balíček
- `kajovochat/main.py` GUI a orchestrace hlasové relace
- `kajovochat/settings.py` minimální produktové nastavení a systémový prompt
- `kajovochat/services/` audio, realtime websocket, OpenAI služby a logování
- `kajovochat/dialogs/` dialog OpenAI a minimální dialog nastavení
- `kajovochat/widgets/` vlastní UI widgety včetně photo-based talking head
- `kajovochat/resources/assets/` obrázkové assety
- `tests/` testy

## Test a kontrola

```bash
python -m compileall -q kajovochat app_gui.py
pytest -q
```

## Omezení

- self-hearing guard je čistě softwarový a bez systémového AEC nelze slíbit absolutní paritu s nativním ChatGPT Voice klientem
- barge-in a anti-echo jsou laděné pro notebookový hands-free provoz, takže při velmi silném odposlechu z reproduktorů může být potřeba provozní test na konkrétním zařízení
