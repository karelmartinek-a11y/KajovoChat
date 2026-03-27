@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\python.exe (
    echo [INFO] Vytvarim virtualni prostredi pro build...
    py -3.11 -m venv .venv 2>nul || py -m venv .venv || python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-build.txt
python tools\build_windows_exe.py
if errorlevel 1 (
    echo [ERROR] Build selhal.
    pause
    exit /b 1
)

echo [INFO] Hotovo. Spustitelny soubor najdes v dist\ChatbotKaja\ChatbotKaja.exe
pause
