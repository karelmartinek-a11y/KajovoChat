@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0

set "APP_NAME=Chatbot Kaja"
set "VENV_DIR=.venv"
set "REQ_FILE=requirements.txt"
set "MARKER=%VENV_DIR%\.kajovochat_requirements_installed"

call :resolve_python
if errorlevel 1 goto :fail

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Vytvarim virtualni prostredi...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 goto :fail

set "INSTALL_DEPS=0"
if not exist "%MARKER%" set "INSTALL_DEPS=1"
if exist "%REQ_FILE%" if "%REQ_FILE%" GTR "%MARKER%" set "INSTALL_DEPS=1"

if "%INSTALL_DEPS%"=="1" (
    echo [INFO] Instaluji nebo aktualizuji zavislosti...
    python -m pip install --upgrade pip
    if errorlevel 1 goto :fail
    python -m pip install -r "%REQ_FILE%"
    if errorlevel 1 goto :fail
    > "%MARKER%" echo installed
)

echo [INFO] Spoustim %APP_NAME%...
python -m kajovochat
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" goto :fail_with_code

goto :eof

:resolve_python
where py >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    %PYTHON_CMD% -V >nul 2>nul
    if errorlevel 1 set "PYTHON_CMD=py"
    exit /b 0
)

where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)

echo [ERROR] Python 3.11+ nebyl nalezen.
echo [ERROR] Nainstaluj Python a potom spust tento soubor znovu.
exit /b 1

:fail_with_code
echo [ERROR] %APP_NAME% skoncila s chybou %EXIT_CODE%.
pause
exit /b %EXIT_CODE%

:fail
echo [ERROR] Nepodarilo se spustit %APP_NAME%.
pause
exit /b 1
