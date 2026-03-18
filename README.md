# Chatbot Kája

Desktop aplikace v `PySide6` pro hlasovou konverzaci přes OpenAI Realtime API.

## Co aplikace umí

- hlasový režim `hands-free` přes klik na měsíc
- režim `push-to-talk` podržením zeměkoule
- výběr modelu, hlasu, jazyka a audio zařízení v GUI
- volitelné technické logování průběhu relace do uživatelského adresáře
- kontrolu integrity binárních assetů při startu

## Požadavky

- Python `3.11` až `3.14`
- funkční mikrofon a reproduktor nebo sluchátka
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

Kompatibilní alternativní vstupní bod:

```bash
python app_gui.py
```

Pomocný Windows skript:

```bat
run_kajovochat.bat
```

## Nastavení

Po spuštění otevřete dialog `OpenAI` a vložte API klíč. Ostatní výchozí parametry lze upravit v dialogu `Nastavení`.

Aplikace standardně funguje jako obecný hlasový asistent podobný ChatGPT. Nepoužívá žádnou pevnou autorizaci uživatele ani individuální scénář.

Konfigurace se ukládá do `settings.json` v uživatelském profilu aplikace. Ve Windows typicky:

```text
C:\Users\<uživatel>\AppData\Local\Kajovo\ChatbotKaja\settings.json
```

OpenAI API klíč je ve Windows uložen přes DPAPI. Na ostatních platformách aplikace používá systémový keyring, pokud je dostupný. Obsah konverzací se do logů zapisuje jen pokud to výslovně povolíte v `Nastavení`.

## Struktura projektu

- `kajovochat/` hlavní aplikační balíček
- `kajovochat/main.py` hlavní okno a orchestrace hlasové relace
- `kajovochat/settings.py` perzistence nastavení a generování systémového promptu
- `kajovochat/services/` OpenAI, audio, realtime websocket a logování
- `kajovochat/dialogs/` dialogy nastavení a API klíče
- `kajovochat/widgets/` vykreslované UI prvky
- `kajovochat/resources/assets/` obrázkové assety
- `tests/` základní smoke testy

## Test a kontrola

```bash
python -m compileall -q kajovochat app_gui.py
pytest -q
```

## Poznámky

- API klíč se ukládá lokálně do uživatelského profilu aplikace.
- Logy relací se ukládají mimo repozitář do uživatelského adresáře nastaveného v aplikaci.
