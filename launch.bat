@echo off
REM Amicus AI — Windows launcher
REM Usage: double-click launch.bat or run from Command Prompt

setlocal EnableDelayedExpansion

REM Change to the directory where this script lives
cd /d "%~dp0"

REM ── Check virtual environment ─────────────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo ERROR: Virtual environment not found.
    echo.
    echo Run the one-time setup first:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo   python -m spacy download en_core_web_lg
    echo.
    pause
    exit /b 1
)

REM ── Activate virtual environment ──────────────────────────────────────────
call venv\Scripts\activate.bat

REM ── Check Ollama ──────────────────────────────────────────────────────────
curl -s --max-time 3 http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama is not responding on http://localhost:11434
    echo.
    echo If Ollama is not running, open it from the Start Menu or system tray.
    echo Or run in a separate window: ollama serve
    echo.
    echo Continuing -- the app will show 'Ollama Not Running' in the sidebar.
    echo.
)

REM ── Open browser after a short delay ─────────────────────────────────────
REM Use PowerShell to open the browser 4 seconds after Streamlit starts
start /b powershell -Command "Start-Sleep 4; Start-Process 'http://localhost:8501'"

REM ── Launch ────────────────────────────────────────────────────────────────
echo Starting Amicus AI at http://localhost:8501 ...
streamlit run app.py

endlocal
