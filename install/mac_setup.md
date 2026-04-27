# Amicus AI — macOS Installation Guide

macOS 12 (Monterey) or later, Intel or Apple Silicon (M1/M2/M3/M4), 8 GB RAM minimum.

---

## Step 1 — Install Homebrew

If Homebrew is not already installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen prompts. After installation, add Homebrew to your PATH if prompted (Apple Silicon Macs require this).

---

## Step 2 — Install pyenv

pyenv lets you install specific Python versions without affecting the system Python.

```bash
brew install pyenv
```

Add pyenv to your shell (add these lines to `~/.zshrc` or `~/.bash_profile`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload your shell:

```bash
source ~/.zshrc
```

---

## Step 3 — Install Python 3.11

```bash
pyenv install 3.11.9
pyenv global 3.11.9
```

Verify:

```bash
python --version
# Python 3.11.9
```

---

## Step 4 — Install Ollama

**Option A — Homebrew:**

```bash
brew install ollama
```

Start the Ollama service:

```bash
brew services start ollama
```

**Option B — Native app:**

Download the macOS app from [ollama.com](https://ollama.com/download/mac) and drag it to Applications. Launch it from the Applications folder — the Ollama icon will appear in your menu bar.

Verify Ollama is running:

```bash
curl http://localhost:11434
# Ollama is running
```

---

## Step 5 — Pull the Language Model

```bash
ollama pull llama3.1:8b
```

Downloads approximately 4.7 GB. On a 100 Mbps connection this takes 5–8 minutes.

---

## Step 6 — Clone the Repository

```bash
git clone https://github.com/your-org/legal-ai-assistant.git
cd legal-ai-assistant
```

---

## Step 7 — Create and Activate the Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Your prompt will change to show `(venv)`.

---

## Step 8 — Install Dependencies

```bash
pip install -r requirements.txt
```

> On Apple Silicon, PyTorch installs the MPS-accelerated build automatically. No extra steps needed.

---

## Step 9 — Download the SpaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

---

## Step 10 — Launch Amicus AI

**Option A — Shell script (easiest):**

```bash
./launch.sh
```

**Option B — Manual:**

```bash
source venv/bin/activate
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

> **Screenshot placeholder** — Amicus AI running in Safari on macOS

---

## Troubleshooting

### "Operation not permitted" when writing to `db/`

Streamlit may need permission to write files if the project is in a restricted location (e.g., Desktop on macOS 13+). Move the project to `~/Documents/legal-ai-assistant` and retry.

### Ollama not running after system restart

If you installed via Homebrew:
```bash
brew services restart ollama
```

If you installed the native app, launch Ollama from Applications.

### M1/M2 — `pip install` fails on a package

Some packages need Rosetta 2 or Xcode Command Line Tools. Install them:

```bash
xcode-select --install
softwareupdate --install-rosetta
```

### Python not found after pyenv install

Ensure pyenv is initialized in your shell profile. Check:

```bash
echo $PYENV_ROOT
which python
```

If `which python` shows `/usr/bin/python` instead of pyenv's path, add the pyenv init lines to your shell profile and reload.

---

## Model Upgrade — 16 GB+ MacBook Pro / Mac Studio / Mac Pro

The sidebar **Model Settings** panel detects your RAM automatically and shows recommended models for your tier.

### 16 GB RAM

```bash
ollama pull llama3.3:8b        # improved instruction following over 3.1
ollama pull mistral-nemo:12b   # better long-document reasoning, 32K context
```

After pulling, open the sidebar → Model Settings → select the model → click **Apply Model**.

### 32 GB+ RAM (Mac Studio, Mac Pro, MacBook Pro M3 Max)

```bash
ollama pull llama3.1:70b   # near GPT-4 quality for complex legal reasoning
```

> Llama 3.1 70B requires approximately 42 GB of unified memory during inference. On a 32 GB system it will use RAM + SSD swap, which is slower. 48 GB+ is recommended for comfortable use.

### Apple Silicon Performance Tips

- Close other applications during long queries to free unified memory.
- Ollama uses Metal (GPU) acceleration on Apple Silicon automatically — no configuration needed.
- The `mistral-nemo:12b` model (32K context window) is particularly well-suited for large deposition transcripts that exceed the 8B model's context.
