#!/usr/bin/env python3
"""
Re-ingestion script — wipe and rebuild ChromaDB from a folder of source documents.

Usage:
    python reingest.py <folder>          # re-ingest all supported files in <folder>
    python reingest.py <folder> --dry-run  # preview without writing

Supported formats: .pdf, .docx, .txt, .xlsx, .xls, .csv

Run this after changing CHUNK_SIZE or CHUNK_OVERLAP in modules/ingestion.py so the
new chunk boundaries take effect. The existing ChromaDB collection is fully wiped
before re-ingestion.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

SUPPORTED = {".pdf", ".docx", ".txt", ".xlsx", ".xls", ".csv"}


def main():
    parser = argparse.ArgumentParser(description="Wipe and rebuild the ChromaDB collection.")
    parser.add_argument("folder", help="Folder containing source documents")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files that would be ingested without writing")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED)

    if not files:
        print(f"No supported files found in '{folder}'.")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED))}")
        sys.exit(0)

    print(f"\nDocuments to ingest ({len(files)} files):")
    for f in files:
        print(f"  {f.name}")

    if args.dry_run:
        print("\nDry run — no changes written.")
        return

    confirm = input(f"\nThis will WIPE the existing ChromaDB collection and re-ingest "
                    f"{len(files)} file(s). Continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    from modules.ingestion import clear_all_documents, ingest_document

    print("\nClearing existing collection...")
    clear_all_documents()
    print("Collection cleared.\n")

    total_chunks = 0
    errors = []

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Ingesting: {file_path.name}")
        try:
            result = ingest_document(str(file_path), original_name=file_path.name)
            total_chunks += result.get("chunks", 0)
            status = result.get("status", "unknown")
            print(f"  → {result.get('pages', '?')} pages, "
                  f"{result.get('chunks', 0)} chunks — {status}")
        except Exception as e:
            print(f"  → ERROR: {e}")
            errors.append((file_path.name, str(e)))
        print()

    print(f"Done. {len(files) - len(errors)}/{len(files)} files ingested, "
          f"{total_chunks} total chunks stored.")
    if errors:
        print("\nFailed files:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
