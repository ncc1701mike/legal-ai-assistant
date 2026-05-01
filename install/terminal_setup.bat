@echo off
:: install/terminal_setup.bat
:: Amicus AI -- first-run installer for Windows.
:: Idempotent: safe to run multiple times; skips steps already completed.
::
:: Usage:
::   Double-click terminal_setup.bat, or run from Command Prompt:
::   cd /d "%~dp0.." && install\terminal_setup.bat [--dry-run] [--simulate-ram N]
::
:: Flags:
::   --dry-run          Simulate the full install without making any changes.
::   --simulate-ram N   Pretend the machine has N GB of RAM for model selection
::                      (useful for testing hardware-tier logic).
::
:: Examples:
::   install\terminal_setup.bat --dry-run
::   install\terminal_setup.bat --dry-run --simulate-ram 8
::   install\terminal_setup.bat --dry-run --simulate-ram 32

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
for %%i in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fi"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "CONFIG_PATH=%PROJECT_DIR%\db\user_config.json"

:: ── Flag parsing ──────────────────────────────────────────────────────────────
set "DRY_RUN=0"
set "SIMULATE_RAM="

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--simulate-ram" (
    set "SIMULATE_RAM=%~2"
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:args_done

:: ── Summary variables (collected throughout, printed at end in dry-run) ───────
set "SUMMARY_PYTHON="
set "SUMMARY_VENV="
set "SUMMARY_PKGS="
set "SUMMARY_OLLAMA_STATUS="
set "SUMMARY_MODEL="
set "SUMMARY_MODEL_SIZE="
set "SUMMARY_MODEL_ACTION="

:: ── Banner ────────────────────────────────────────────────────────────────────
echo.
echo  ==============================
echo   Amicus AI -- Setup
echo   Privacy-first legal document analysis
echo  ==============================
echo.

if "!DRY_RUN!"=="1" (
    echo  **********************************************************
    echo   DRY RUN MODE -- No changes will be made to your system
    if defined SIMULATE_RAM (
        echo   Simulating RAM: !SIMULATE_RAM! GB
    )
    echo  **********************************************************
    echo.
)


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
    set "SUMMARY_PYTHON=Python 3.11+ not found -- install required"
    if "!DRY_RUN!"=="1" (
        echo [DRY RUN] A real install would exit here. Continuing simulation.
        set "PY_VER=(not found)"
        goto :step2
    )
    pause
    exit /b 1
)

echo [amicus] Found Python !PY_VER! (!PYTHON_CMD!)
set "SUMMARY_PYTHON=Python !PY_VER! (!PYTHON_CMD!)"


:: ── Step 2: Virtual environment ───────────────────────────────────────────────
:step2
echo.
echo ── Step 2/5 -- Setting up virtual environment ───────────────────────────────
echo.

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [amicus] Virtual environment already exists -- skipping creation
    set "SUMMARY_VENV=venv\ already exists (would skip)"
) else (
    if "!DRY_RUN!"=="1" (
        echo [DRY RUN] Would run: !PYTHON_CMD! -m venv "%VENV_DIR%"
        set "SUMMARY_VENV=venv\ (would be created at %VENV_DIR%)"
    ) else (
        echo [amicus] Creating virtual environment in %VENV_DIR% ...
        !PYTHON_CMD! -m venv "%VENV_DIR%"
        if !errorlevel! neq 0 (
            echo [amicus] ERROR: Failed to create virtual environment.
            pause
            exit /b 1
        )
        echo [amicus] Virtual environment created
        set "SUMMARY_VENV=venv\ created at %VENV_DIR%"
    )
)

if "!DRY_RUN!"=="1" (
    echo [DRY RUN] Would run: call "%VENV_DIR%\Scripts\activate.bat"
) else (
    call "%VENV_DIR%\Scripts\activate.bat"
    echo [amicus] Activated: %VENV_DIR%
)


:: ── Step 3: Python dependencies ───────────────────────────────────────────────
echo.
echo ── Step 3/5 -- Installing Python dependencies ───────────────────────────────
echo.

set "REQUIREMENTS=%PROJECT_DIR%\requirements.txt"
if not exist "%REQUIREMENTS%" (
    echo [amicus] ERROR: requirements.txt not found at %REQUIREMENTS%
    set "SUMMARY_PKGS=requirements.txt not found"
    if "!DRY_RUN!"=="1" (
        echo [DRY RUN] A real install would exit here. Continuing simulation.
        goto :step4
    )
    pause
    exit /b 1
)

:: Count packages (read-only -- always runs)
set "PKG_COUNT=0"
for /f "tokens=*" %%L in (%REQUIREMENTS%) do (
    set /a PKG_COUNT+=1
)
set "SUMMARY_PKGS=!PKG_COUNT! packages from requirements.txt"

if "!DRY_RUN!"=="1" (
    echo [DRY RUN] Would run: pip install --quiet --upgrade pip
    echo [DRY RUN] Would run: pip install --quiet -r "%REQUIREMENTS%"
    echo [amicus] Would install !SUMMARY_PKGS!
) else (
    echo [amicus] Running pip install (this may take a few minutes on first run)...
    pip install --quiet --upgrade pip
    pip install --quiet -r "%REQUIREMENTS%"
    if !errorlevel! neq 0 (
        echo [amicus] ERROR: pip install failed. Check your internet connection and try again.
        pause
        exit /b 1
    )
    echo [amicus] Python dependencies installed
)


:: ── Step 4: Ollama ────────────────────────────────────────────────────────────
:step4
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
    set "SUMMARY_OLLAMA_STATUS=NOT INSTALLED -- install required"
    if "!DRY_RUN!"=="1" (
        echo [DRY RUN] A real install would exit here. Continuing simulation.
        goto :step5
    )
    pause
    exit /b 1
)

echo [amicus] Ollama found

:: Health check -- always runs (read-only, shows real Ollama status)
curl -sf http://localhost:11434/api/tags >nul 2>&1
if !errorlevel! == 0 (
    echo [amicus] Ollama is already running
    set "SUMMARY_OLLAMA_STATUS=running"
) else (
    set "SUMMARY_OLLAMA_STATUS=not running"
    if "!DRY_RUN!"=="1" (
        echo [amicus] Ollama is not running
        echo [DRY RUN] Would run: start /b "" ollama serve
    ) else (
        echo [amicus] Ollama is not running -- attempting to start...
        start /b "" ollama serve >nul 2>&1

        :: Wait up to 10 seconds for Ollama to become available
        set "OLLAMA_READY=0"
        for %%i in (1 2 3 4 5) do (
            if "!OLLAMA_READY!" == "0" (
                timeout /t 2 /nobreak >nul
                curl -sf http://localhost:11434/api/tags >nul 2>&1
                if !errorlevel! == 0 set "OLLAMA_READY=1"
            )
        )

        if "!OLLAMA_READY!" == "1" (
            echo [amicus] Ollama started successfully
        ) else (
            echo [amicus] WARNING: Could not start Ollama automatically.
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
    )
)


:: ── Step 5: Select and pull the right model ───────────────────────────────────
:step5
echo.
echo ── Step 5/5 -- Selecting and downloading your Analysis Engine ────────────────
echo.

:: RAM detection -- always runs (read-only); overridden by --simulate-ram
if defined SIMULATE_RAM (
    set "RAM_GB=!SIMULATE_RAM!"
    echo [amicus] Simulated RAM: !RAM_GB! GB (--simulate-ram^)
) else (
    for /f "tokens=*" %%r in ('python -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))" 2^>nul') do (
        set "RAM_GB=%%r"
    )
    if not defined RAM_GB set "RAM_GB=8"
    echo [amicus] Detected RAM: !RAM_GB! GB
)

:: Windows stays conservative regardless of RAM
set "SELECTED_MODEL=llama3.1:8b"
set "MODEL_SIZE=4.7"
set "PROFILE_NAME=Standard"

if !RAM_GB! GEQ 24 (
    echo [amicus] NOTE: 24+ GB detected. Using Standard profile (Windows conservative default).
    echo          For Enhanced profile, ask your IT administrator to run: ollama pull llama3.3:8b
)

echo [amicus] Selected profile: !PROFILE_NAME! (!SELECTED_MODEL!)
set "SUMMARY_MODEL=!SELECTED_MODEL! (!PROFILE_NAME!)"
set "SUMMARY_MODEL_SIZE=!MODEL_SIZE! GB"

:: Check if already installed (read-only -- runs if Ollama available)
where ollama >nul 2>&1
if !errorlevel! == 0 (
    ollama list 2>nul | findstr /b "!SELECTED_MODEL!" >nul 2>&1
    if !errorlevel! == 0 (
        if "!DRY_RUN!"=="1" (
            echo [amicus] Model !SELECTED_MODEL! is already installed -- would skip download
            set "SUMMARY_MODEL_ACTION=already installed (no download needed)"
        ) else (
            echo [amicus] Model !SELECTED_MODEL! is already installed -- skipping download
        )
        goto :write_config
    )
)

set "SUMMARY_MODEL_ACTION=download !MODEL_SIZE! GB (one time only)"
if "!DRY_RUN!"=="1" (
    echo [DRY RUN] Would run: ollama pull !SELECTED_MODEL! (!MODEL_SIZE! GB download^)
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
:write_config
if "!DRY_RUN!"=="1" (
    echo [DRY RUN] Would write: db\user_config.json  (primary_model: !SELECTED_MODEL!^)
    goto :summary
)

if not exist "%PROJECT_DIR%\db" mkdir "%PROJECT_DIR%\db"

if exist "%CONFIG_PATH%" (
    python -c "import json, pathlib; p = pathlib.Path(r'%CONFIG_PATH%'); cfg = {}; [cfg.update(json.loads(p.read_text()))] if p.exists() else None; cfg['primary_model'] = '!SELECTED_MODEL!'; p.write_text(json.dumps(cfg, indent=2))"
) else (
    echo {"primary_model": "!SELECTED_MODEL!"} > "%CONFIG_PATH%"
)

echo [amicus] Model choice saved to db\user_config.json
goto :done


:: ── Dry-run summary ───────────────────────────────────────────────────────────
:summary
echo.
echo  -- Dry Run Summary -----------------------------------------------
echo   Detected RAM:    !RAM_GB! GB
if defined SIMULATE_RAM echo                    (simulated via --simulate-ram)
echo   Detected OS:     Windows
echo   Selected model:  !SUMMARY_MODEL!
echo   Model action:    !SUMMARY_MODEL_ACTION!
echo   Ollama status:   !SUMMARY_OLLAMA_STATUS!
echo   Would download:  !SUMMARY_MODEL_SIZE!
echo   Would write:     db\user_config.json
echo   Would create:    !SUMMARY_VENV!
echo   Would install:   !SUMMARY_PKGS!
echo  ------------------------------------------------------------------
echo   Run without --dry-run to perform actual installation.
echo.
goto :end


:: ── Done ──────────────────────────────────────────────────────────────────────
:done
echo.
echo  ============================================================
echo   Amicus is ready.
echo  ============================================================
echo.
echo   Run:  streamlit run app.py
echo.
pause

:end
endlocal
