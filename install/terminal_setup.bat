@echo off
:: install/terminal_setup.bat
:: Amicus AI -- first-run installer for Windows.
:: Idempotent: safe to run multiple times; skips steps already completed.
::
:: Usage:
::   Double-click terminal_setup.bat, or run from Command Prompt:
::   cd /d "%~dp0.." && install\terminal_setup.bat

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
for %%i in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fi"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "CONFIG_PATH=%PROJECT_DIR%\db\user_config.json"

echo.
echo  ==============================
echo   Amicus AI -- Setup
echo   Privacy-first legal document analysis
echo  ==============================
echo.


:: ── Step 1: Python 3.11+ ──────────────────────────────────────────────────────
echo.
echo ── Step 1/5 -- Checking Python version ──────────────────────────────────────
echo.

set "PYTHON_CMD="
for %%c in (python3.11 python3.12 python3.13 python3 python) do (
    if not defined PYTHON_CMD (
        where %%c >nul 2>&1
        if !errorlevel! == 0 (
            for /f "tokens=*" %%v in ('%%c -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2^>nul') do (
                set "PY_VER=%%v"
            )
            for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
                set "PY_MAJOR=%%a"
                set "PY_MINOR=%%b"
            )
            if !PY_MAJOR! GEQ 3 (
                if !PY_MINOR! GEQ 11 (
                    set "PYTHON_CMD=%%c"
                )
            )
        )
    )
)

if not defined PYTHON_CMD (
    echo [amicus] ERROR: Python 3.11 or newer is required but was not found.
    echo.
    echo   Download Python from: https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [amicus] Found Python !PY_VER! (!PYTHON_CMD!)


:: ── Step 2: Virtual environment ───────────────────────────────────────────────
echo.
echo ── Step 2/5 -- Setting up virtual environment ───────────────────────────────
echo.

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [amicus] Virtual environment already exists -- skipping creation
) else (
    echo [amicus] Creating virtual environment in %VENV_DIR% ...
    !PYTHON_CMD! -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [amicus] ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [amicus] Virtual environment created
)

call "%VENV_DIR%\Scripts\activate.bat"
echo [amicus] Activated: %VENV_DIR%


:: ── Step 3: Python dependencies ───────────────────────────────────────────────
echo.
echo ── Step 3/5 -- Installing Python dependencies ───────────────────────────────
echo.

set "REQUIREMENTS=%PROJECT_DIR%\requirements.txt"
if not exist "%REQUIREMENTS%" (
    echo [amicus] ERROR: requirements.txt not found at %REQUIREMENTS%
    pause
    exit /b 1
)

echo [amicus] Running pip install (this may take a few minutes on first run)...
pip install --quiet --upgrade pip
pip install --quiet -r "%REQUIREMENTS%"
if !errorlevel! neq 0 (
    echo [amicus] ERROR: pip install failed. Check your internet connection and try again.
    pause
    exit /b 1
)
echo [amicus] Python dependencies installed


:: ── Step 4: Ollama ────────────────────────────────────────────────────────────
echo.
echo ── Step 4/5 -- Checking Ollama ──────────────────────────────────────────────
echo.

where ollama >nul 2>&1
if !errorlevel! neq 0 (
    echo [amicus] ERROR: Ollama is not installed.
    echo.
    echo   Install Ollama:
    echo     Go to https://ollama.ai and click "Download for Windows"
    echo     Run the installer, then run this script again.
    echo.
    pause
    exit /b 1
)

echo [amicus] Ollama found

:: Check Ollama is running
curl -sf http://localhost:11434/api/tags >nul 2>&1
if !errorlevel! neq 0 (
    echo [amicus] WARNING: Ollama does not appear to be running.
    echo.
    echo   Look for the llama icon in your system tray (bottom-right of your screen).
    echo   Click it to open Ollama. If you don't see it, open Ollama from the Start menu.
    echo.
    set /p CONTINUE="  Continue anyway? [y/N] "
    if /i not "!CONTINUE!" == "y" (
        echo   Exiting. Start Ollama and run this script again.
        pause
        exit /b 0
    )
)


:: ── Step 5: Select and pull the right model ───────────────────────────────────
echo.
echo ── Step 5/5 -- Selecting and downloading your Analysis Engine ────────────────
echo.

:: Detect RAM in GB via Python
for /f "tokens=*" %%r in ('python -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))" 2^>nul') do (
    set "RAM_GB=%%r"
)
if not defined RAM_GB set "RAM_GB=8"
echo [amicus] Detected RAM: !RAM_GB! GB

:: Windows stays conservative regardless of RAM
:: (Windows has higher OS memory overhead than macOS)
set "SELECTED_MODEL=llama3.1:8b"
set "MODEL_SIZE=4.7"
set "PROFILE_NAME=Standard"

if !RAM_GB! GEQ 24 (
    echo [amicus] NOTE: 24+ GB detected. Using Standard profile (Windows conservative default).
    echo          For Enhanced profile, ask your IT administrator to run: ollama pull llama3.3:8b
)

echo [amicus] Selected profile: !PROFILE_NAME! (!SELECTED_MODEL!)

:: Check if already installed
ollama list 2>nul | findstr /b "!SELECTED_MODEL!" >nul 2>&1
if !errorlevel! == 0 (
    echo [amicus] Model !SELECTED_MODEL! is already installed -- skipping download
) else (
    echo [amicus] Downloading your Analysis Engine (!MODEL_SIZE! GB -- one time only)...
    echo   This may take 10-20 minutes depending on your internet speed.
    echo.
    ollama pull !SELECTED_MODEL!
    if !errorlevel! neq 0 (
        echo [amicus] ERROR: Model download failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
    echo [amicus] Model !SELECTED_MODEL! downloaded successfully
)


:: ── Write model choice to config ──────────────────────────────────────────────
if not exist "%PROJECT_DIR%\db" mkdir "%PROJECT_DIR%\db"

if exist "%CONFIG_PATH%" (
    python -c "import json, pathlib; p = pathlib.Path(r'%CONFIG_PATH%'); cfg = {}; [cfg.update(json.loads(p.read_text()))] if p.exists() else None; cfg['primary_model'] = '!SELECTED_MODEL!'; p.write_text(json.dumps(cfg, indent=2))"
) else (
    echo {"primary_model": "!SELECTED_MODEL!"} > "%CONFIG_PATH%"
)

echo [amicus] Model choice saved to db\user_config.json


:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Amicus is ready.
echo  ============================================================
echo.
echo   Run:  streamlit run app.py
echo.
pause
endlocal
