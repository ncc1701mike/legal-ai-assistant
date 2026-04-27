# Amicus AI — Windows Installation Guide

Windows 10 (version 21H2+) or Windows 11 (64-bit), 8 GB RAM minimum.

---

## Step 1 — Install Python 3.11

1. Download Python 3.11 from [python.org/downloads](https://www.python.org/downloads/).
   - Use the **Windows installer (64-bit)**.
2. Run the installer. On the first screen, check **"Add python.exe to PATH"** before clicking Install Now.

   > **Screenshot placeholder** — installer first screen with PATH checkbox highlighted

3. Verify the install:
   ```cmd
   python --version
   ```
   Expected output: `Python 3.11.x`

---

## Step 2 — Install Git (if not already installed)

Download from [git-scm.com](https://git-scm.com/download/win) and run the installer with default settings.

---

## Step 3 — Install Ollama for Windows

1. Download the Ollama Windows installer from [ollama.com](https://ollama.com/download/windows).
2. Run `OllamaSetup.exe`. Ollama installs as a background service and starts automatically.

   > **Screenshot placeholder** — Ollama system tray icon in taskbar

3. Verify Ollama is running — open a browser and visit `http://localhost:11434`. You should see `Ollama is running`.

---

## Step 4 — Pull the Language Model

Open **Command Prompt** or **PowerShell** and run:

```cmd
ollama pull llama3.1:8b
```

This downloads approximately 4.7 GB. Time depends on your internet connection.

> For 16 GB+ systems, you may also pull a higher-quality model. See the Model Upgrade section at the bottom.

---

## Step 5 — Clone the Repository

```cmd
git clone https://github.com/your-org/legal-ai-assistant.git
cd legal-ai-assistant
```

---

## Step 6 — Create the Virtual Environment

```cmd
python -m venv venv
```

---

## Step 7 — Activate the Virtual Environment

```cmd
venv\Scripts\activate
```

Your prompt should change to show `(venv)`.

---

## Step 8 — Install Dependencies

```cmd
pip install -r requirements.txt
```

This installs all Python packages including PyTorch, ChromaDB, Streamlit, and LangChain. Expect 3–5 minutes on first run.

---

## Step 9 — Download the SpaCy Language Model

```cmd
python -m spacy download en_core_web_lg
```

---

## Step 10 — Launch Amicus AI

**Option A — Double-click launcher (easiest):**

Double-click `launch.bat` in the project folder.

**Option B — From the terminal:**

```cmd
venv\Scripts\activate
streamlit run app.py
```

Your browser should open automatically at `http://localhost:8501`.

> **Screenshot placeholder** — Amicus AI onboarding screen in browser

---

## Troubleshooting

### Windows Defender flags Ollama

Ollama may trigger a Windows Defender SmartScreen warning on first run because it is not yet widely signed. Click **"More info"** then **"Run anyway"**. Ollama is open-source: [github.com/ollama/ollama](https://github.com/ollama/ollama).

### Firewall blocking port 11434

If Streamlit loads but shows "Ollama Not Running":

1. Open **Windows Defender Firewall** → **Allow an app or feature through Windows Defender Firewall**.
2. Click **Change settings**, then **Allow another app**.
3. Browse to `C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe` and add it.
4. Alternatively, allow the port directly:
   ```powershell
   netsh advfirewall firewall add rule name="Ollama" dir=in action=allow protocol=TCP localport=11434
   ```

### Path length limit (error during pip install)

Windows limits file paths to 260 characters by default. If you see path-length errors:

1. Open **Group Policy Editor** (`gpedit.msc`).
2. Navigate to: **Local Computer Policy → Computer Configuration → Administrative Templates → System → Filesystem**.
3. Enable **"Enable Win32 long paths"**.

Or via PowerShell (run as Administrator):
```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
```

### `venv\Scripts\activate` gives a permission error

Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### pip install fails on `torch`

PyTorch wheels for Windows are large (~2 GB). If the download times out:
```cmd
pip install torch --timeout 300
pip install -r requirements.txt
```

---

## Model Upgrade (16 GB+ RAM)

Inside Amicus, open the sidebar → **Model Settings**. The recommended models for your hardware tier are shown automatically. For systems with 16 GB RAM:

| Model | RAM Required | Download |
|---|---|---|
| Llama 3.3 8B | 8 GB | `ollama pull llama3.3:8b` |
| Mistral Nemo 12B | 10 GB | `ollama pull mistral-nemo:12b` |

After pulling, select the model in the sidebar and click **Apply Model**.
