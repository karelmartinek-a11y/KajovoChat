@echo off
set REPO=C:\github\KajovoChat
call "%REPO%\.venv\Scripts\activate.bat"
pushd "%REPO%"
python -m kajovochat
popd
