@echo off
cd /d "%~dp0"
if not exist "backend\venv\Scripts\python.exe" (
  echo SELENE-REG virtual environment not found.
  echo Create it with: python -m venv backend\venv
  echo Then install: backend\venv\Scripts\pip.exe install -r backend\requirements.txt
  pause
  exit /b 1
)
start "" "http://127.0.0.1:8000"
backend\venv\Scripts\python.exe serve.py
pause
