# Repository Guidelines

## Jazyk komunikace

- AI agent komunikuje s uživatelem výhradně česky, pokud uživatel výslovně neurčí jinak.
- Dokumentace, poznámky a komentáře v kódu se píšou česky.
- Zdrojový kód, názvy proměnných, funkcí a API mohou zůstat v angličtině.

## Kódování souborů

- Všechny textové soubory musí být v `UTF-8 bez BOM`.

## Aktuální struktura projektu

- `kajovochat/` hlavní aplikace
- `kajovochat/main.py` GUI a řízení hlasové relace
- `kajovochat/settings.py` nastavení aplikace a systémový prompt
- `kajovochat/services/` OpenAI služby, audio, realtime komunikace a logování
- `kajovochat/dialogs/` dialogy pro nastavení a API klíč
- `kajovochat/widgets/` vlastní UI widgety
- `kajovochat/resources/assets/` assety pro GUI
- `app_gui.py` kompatibilní alternativní vstupní bod
- `run_kajovochat.bat` pomocný start skript pro Windows
- `tests/` testy

## Spuštění

- Vytvoření prostředí: `python -m venv .venv`
- Aktivace ve Windows: `.\\.venv\\Scripts\\activate`
- Instalace závislostí: `pip install -r requirements.txt`
- Hlavní spuštění: `python -m kajovochat`
- Alternativní spuštění: `python app_gui.py`

## Styl kódu

- Cílová verze Pythonu je `3.11+`.
- Preferují se typové anotace a `f-string`.
- Funkce a proměnné mají být `snake_case`, třídy `PascalCase`.
- Každá změna má držet skutečný stav aplikace, bez historických nebo cizích artefaktů.

## Testování

- Používej `pytest`.
- Testy patří do `tests/` a mají se jmenovat `test_*.py`.
- Před odevzdáním změn spusť aspoň:
  - `python -m compileall -q kajovochat app_gui.py`
  - `pytest -q`

## Bezpečnost a provoz

- Do repozitáře nepatří provozní logy, cache, `__pycache__`, exporty dat ani lokální audity.
- Do repozitáře nepatří API klíče ani jiné tajné údaje.
- Repozitář má obsahovat jen zdrojový kód, potřebné assety, relevantní testy a aktuální dokumentaci.
