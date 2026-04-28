# Amicus AI — Quick Start Guide

**For attorneys and legal staff. No technical knowledge required.**

Amicus analyzes your legal documents privately on your computer. Nothing you upload is ever sent to the internet — everything stays on your machine.

---

## What you'll need

- A Mac or Windows computer (laptop or desktop)
- An internet connection for the one-time setup
- About 10–20 minutes for the initial installation

After setup, Amicus works completely offline.

---

## Step 1 — Install Ollama

Ollama is the free AI engine that powers Amicus. It runs privately on your computer.

**On Mac:**

1. Go to **[ollama.ai](https://ollama.ai)** in your web browser
2. Click **Download for Mac**
3. Open the downloaded file and drag Ollama to your Applications folder
4. Open Ollama from your Applications folder or Launchpad
5. You'll see a small llama icon appear in your menu bar (top right of your screen) — that means it's running

**On Windows:**

1. Go to **[ollama.ai](https://ollama.ai)** in your web browser
2. Click **Download for Windows**
3. Run the downloaded installer and follow the prompts
4. After installation, Ollama starts automatically. Look for the llama icon near the clock in your system tray (bottom right of your screen)

---

## Step 2 — Run the Amicus installer

Open a terminal (Mac) or Command Prompt (Windows) and run the installer script. It will automatically set up Python, install dependencies, and download the right AI model for your computer.

**On Mac:**

Open the Terminal app (search "Terminal" in Spotlight) and run:

```
cd /path/to/amicus
chmod +x install/terminal_setup.sh
./install/terminal_setup.sh
```

**On Windows:**

Open Command Prompt and run:

```
cd C:\path\to\amicus
install\terminal_setup.bat
```

Or simply double-click `terminal_setup.bat` in File Explorer.

The installer will:

1. Check your Python version
2. Create an isolated Python environment
3. Install all required software
4. Verify Ollama is running
5. Download the right AI model for your computer (about 5 GB — one time only, takes 10–20 minutes)

When it finishes you'll see: **Amicus is ready.**

---

## Launch Amicus

After setup completes, start Amicus by running:

```
streamlit run app.py
```

This opens Amicus in your web browser. You're ready to start working.

**To launch Amicus in the future**, just run `streamlit run app.py` from the Amicus folder. You don't need to run the installer again.

---

## What you can do in Amicus

- **Upload documents** — drag and drop PDFs, Word documents, or text files into the sidebar
- **Ask questions** — type natural language questions about your documents in the chat area
- **Summarize documents** — get structured summaries of any document or your entire case file
- **Redact PII** — automatically remove names, dates, case numbers, and other identifying information

---

## Frequently asked questions

**Do my documents leave my computer?**
No. Everything in Amicus runs locally on your machine. Your documents, queries, and analysis results never leave your computer and are never sent to any server.

**Do I need to be connected to the internet to use Amicus?**
Only during the one-time setup. After that, Amicus works completely offline — even in areas with no internet access.

**The sidebar shows "Not responding" next to the AI engine.**
Ollama isn't running. On Mac, open Ollama from your Applications folder — you'll see the llama icon appear in your menu bar. On Windows, look for the llama icon in your system tray (bottom right); if you don't see it, open Ollama from the Start menu.

**What if the download fails or is very slow?**
Make sure you have a stable internet connection and run the installer script again — it will pick up where it left off. The download is about 5 GB; on a slow connection it may take 30–60 minutes. Once it completes, you won't need to download it again.

**Who do I contact for help?**
Contact your IT administrator. For technical details, refer to `install/terminal_setup.sh` (Mac) or `install/terminal_setup.bat` (Windows).

---

*Amicus AI — Private legal intelligence. Built for litigation attorneys.*
