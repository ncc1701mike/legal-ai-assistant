#!/usr/bin/env python3
# sync/sync_feedback.py
# Daily feedback sync to Google Drive
# Scheduled via launchd — runs at 9pm MST every day
# Install: python3 sync/sync_feedback.py --install-launchd

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = Path.home() / "amicus_sync.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
INSTALL_DIR = Path.home() / "AIE9" / "legal-ai-assistant"
FEEDBACK_FILE = INSTALL_DIR / "feedback" / "feedback.jsonl"
CREDENTIALS_FILE = INSTALL_DIR / "credentials.json"
TOKEN_FILE = INSTALL_DIR / "feedback" / ".gdrive_token.json"
GDRIVE_FOLDER_NAME = "Amicus Beta Feedback"

# ── Google Drive sync ─────────────────────────────────────────────────────────
def get_drive_service():
    """Authenticate and return Google Drive service."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError:
        logger.error("Google API packages not installed. Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        sys.exit(1)

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                logger.error(f"credentials.json not found at {CREDENTIALS_FILE}")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, folder_name: str) -> str:
    """Get the Amicus Beta Feedback folder ID, creating it if needed."""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        folder_id = files[0]["id"]
        logger.info(f"Found existing Drive folder: {folder_name} ({folder_id})")
        return folder_id

    # Create it
    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    folder_id = folder["id"]
    logger.info(f"Created Drive folder: {folder_name} ({folder_id})")
    return folder_id


def get_user_id() -> str:
    """Read the stable user ID from the feedback directory."""
    uid_file = INSTALL_DIR / "feedback" / ".user_id"
    if uid_file.exists():
        return uid_file.read_text().strip()
    return "unknown_user"


def sync_feedback():
    """Main sync function — upload today's feedback to Google Drive."""
    if not FEEDBACK_FILE.exists():
        logger.info("No feedback file found — nothing to sync.")
        return

    # Count entries
    entry_count = 0
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry_count += 1

    if entry_count == 0:
        logger.info("Feedback file is empty — nothing to sync.")
        return

    logger.info(f"Syncing {entry_count} feedback entries to Google Drive...")

    try:
        from googleapiclient.http import MediaFileUpload

        service = get_drive_service()
        folder_id = get_or_create_folder(service, GDRIVE_FOLDER_NAME)

        user_id = get_user_id()
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"feedback_{user_id}_{date_str}.jsonl"

        # Check if file already exists for today (update instead of duplicate)
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        existing = service.files().list(q=query, fields="files(id)").execute()
        existing_files = existing.get("files", [])

        media = MediaFileUpload(str(FEEDBACK_FILE), mimetype="application/json", resumable=True)

        if existing_files:
            # Update existing file
            file_id = existing_files[0]["id"]
            service.files().update(fileId=file_id, media_body=media).execute()
            logger.info(f"Updated existing Drive file: {filename}")
        else:
            # Create new file
            metadata = {"name": filename, "parents": [folder_id]}
            service.files().create(body=metadata, media_body=media, fields="id").execute()
            logger.info(f"Created new Drive file: {filename}")

        logger.info(f"Sync complete — {entry_count} entries uploaded as {filename}")

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)


# ── launchd installer ─────────────────────────────────────────────────────────
def install_launchd():
    """Install the launchd plist to run sync daily at 9pm MST (04:00 UTC)."""
    user_id = get_user_id()
    plist_label = "com.amicusai.feedback.sync"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{plist_label}.plist"

    python_bin = sys.executable
    script_path = INSTALL_DIR / "sync" / "sync_feedback.py"
    log_path = Path.home() / "amicus_sync.log"

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_bin}</string>
        <string>{script_path}</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>4</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>RunAtLoad</key>
    <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>"""

    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)
    logger.info(f"launchd plist written to {plist_path}")

    # Load it
    os.system(f"launchctl unload {plist_path} 2>/dev/null; launchctl load {plist_path}")
    logger.info("launchd agent loaded — sync will run daily at 9pm MST")
    print(f"\n✔  Feedback sync scheduled daily at 9:00 PM MST")
    print(f"   Plist: {plist_path}")
    print(f"   Log:   {log_path}\n")


def uninstall_launchd():
    """Remove the launchd agent."""
    plist_label = "com.amicusai.feedback.sync"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{plist_label}.plist"
    os.system(f"launchctl unload {plist_path} 2>/dev/null")
    if plist_path.exists():
        plist_path.unlink()
    print("✔  Feedback sync agent removed.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amicus AI feedback sync")
    parser.add_argument("--install-launchd", action="store_true", help="Install daily launchd sync agent")
    parser.add_argument("--uninstall-launchd", action="store_true", help="Remove launchd sync agent")
    parser.add_argument("--test", action="store_true", help="Run sync immediately (test mode)")
    args = parser.parse_args()

    if args.install_launchd:
        install_launchd()
    elif args.uninstall_launchd:
        uninstall_launchd()
    else:
        sync_feedback()
