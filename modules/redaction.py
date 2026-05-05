# modules/redaction.py
# PII detection and redaction using spaCy NER + regex patterns
# Turns real case documents into clean, reusable templates

import re
import spacy
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.getLogger("spacy").setLevel(logging.ERROR)

def is_pdf_binary(file_path: str) -> bool:
    """Detect if a file contains PDF binary regardless of extension."""
    try:
        with open(file_path, "rb") as f:
            header = f.read(5)
            return header == b'%PDF-'
    except Exception:
        return False

# ── Load spaCy Model ──────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise OSError("spaCy model not found. Run: python -m spacy download en_core_web_lg")


# ── Placeholder Map ───────────────────────────────────────────────────────────
ENTITY_PLACEHOLDERS = {
    "PERSON":     "[PARTY_NAME]",
    "ORG":        "[ORGANIZATION]",
    "GPE":        "[LOCATION]",
    "LOC":        "[LOCATION]",
    "DATE":       "[DATE]",
    "TIME":       "[TIME]",
    "CARDINAL":   "[NUMBER]",
    "LAW":        "[LEGAL_REFERENCE]",
    "NORP":       "[GROUP]",
}

# ── Regex Patterns ────────────────────────────────────────────────────────────
REGEX_PATTERNS = [
    # SSN with label
    (r'\bSSN:?\s*\d{3}-\d{2}-\d{4}\b',                           "[SSN]"),
    # SSN — must come before generic number patterns
    (r'\b\d{3}-\d{2}-\d{4}\b',                                   "[SSN]"),
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',    "[EMAIL]"),
    # Phone numbers — must have separators, not solid digit blocks
    (r'\b(\+1[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b',      "[PHONE]"),
    # International phone numbers with spaces — +44 20 7946 0958
    (r'\+\d{1,3}[\s.-]\d{1,4}[\s.-]\d{3,4}[\s.-]\d{3,4}\b',     '[PHONE]'),
    # Bar numbers — must come before case numbers
    (r'\bBar\s+No\.?\s*\d+',                                      "[BAR_NO]"),
    # Bates numbers — alphanumeric with hyphens or underscores
    (r'\b[A-Z]{2,}[_-][A-Z0-9]{2,}[_-]\d{4,}\b',                '[BATES_NO]'),
    # Docket numbers
    (r'\bDocket\s+No\.?\s*[\w\-]+',                               "[DOCKET_NO]"),
    # Case numbers — specific formats
    (r'\bCase\s+No\.?\s*[\w\-]+',                                 "[CASE_NO]"),
    (r'\bNo\.\s*\d[\w\-]*',                                       "[CASE_NO]"),
    # Alphanumeric case numbers like 24-CV-00123
    (r'\b\d{2,4}-[A-Z]{2,}-\d{4,}\b',                            "[CASE_NO]"),
    # Pure numeric case numbers like 123-456 — negative lookahead prevents
    # partial match of EEOC-style NNN-YYYY-NNNNN numbers
    (r'\b\d{1,3}-\d{3,6}\b(?!-\d)',                               "[CASE_NO]"),
    # Street addresses — permissive mid-line variant first (1-4 words between
    # number and street type, any capitalisation)
    (r'\b\d{3,5}[ \t]+(?:[A-Za-z]+[ \t]+){1,4}(?:Street|Ave(?:nue)?|Blvd|Boulevard|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Circle|Cir|Trail|Terrace|Ter|Highway|Hwy)(?:\.|\b)(?:,?[ \t]+(?:Suite|Ste|Apt|Unit|Floor|Fl)\.?[ \t]*[\w]+)?',
     '[STREET_ADDRESS]'),
    # "The <Name>" named streets — 1900 The Alameda
    (r'\b\d{3,5}\s+The\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:,?\s+(?:Suite|Ste|Apt|Unit|Floor|Fl)\.?\s*[\w]+)?',
     '[STREET_ADDRESS]'),
    # ZIP codes — only match in address context (after comma+space or space after state)
    (r'(?<=,\s)\d{5}(?!\d)',                                       "[ZIP_CODE]"),
    (r'(?<=\s)[A-Z]{2}\s\d{5}(?!\d)',                             "[ZIP_CODE]"),
    # Leading parenthesis before capitalized name/org — (Wayland Station...
    (r'\(([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)',                 '[ORGANIZATION]'),
    # All-caps court headers — IN THE THIRD DISTRICT COURT
    (r'\bIN\s+THE\s+[A-Z][A-Z\s]+(?:COURT|TRIBUNAL|CIRCUIT|DISTRICT|DIVISION)\b',
     "[ORGANIZATION]"),
    # All-caps org suffixes — NEXAGEN INC, WAYLAND STATION LLC, etc.
    (r'\b(?:[A-Z]{2,}\s+){1,6}(?:INC|LLC|LLP|CORP|CO|COMPANY|ASSOCIATES|GROUP|INSTITUTE|FOUNDATION|AUTHORITY|AGENCY|SERVICES|SOLUTIONS)\b',
     "[ORGANIZATION]"),
    # US State names standalone
    (r'\b(Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|'
     r'Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|'
     r'Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|'
     r'Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|'
     r'New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|'
     r'Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|'
     r'Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b',
     "[LOCATION]"),
]


def redact_text(text: str, aggressive: bool = False,
                categories: List[str] = None) -> Tuple[str, Dict]:
    """
    Redact PII from text using spaCy NER + regex patterns.

    Args:
        text: Input text to redact
        aggressive: If True, also redacts CARDINAL numbers and LAW references
        categories: List of placeholder types to redact e.g. ["[SSN]", "[PARTY_NAME]"]
                   If None, redacts all categories.

    Returns:
        Tuple of (redacted_text, redaction_report)
    """
    redaction_log = []
    redacted = text

    # ── Step -1: Protect named party/entity terms that must never be redacted ──
    # These are replaced with unique sentinels before ANY other pass runs and
    # restored at the very end — guaranteeing they survive regex, NER, city,
    # name-sweep, and compound-merge passes.
    # Sentinel format [[prot_N]]: brackets trigger the NER guard; lowercase
    # letters are immune to all uppercase-only regex patterns (Bates, all-caps
    # org, court-header, etc.).
    PROTECTED_TERMS = [
        "Nexagen Pharmaceuticals, Inc.",
        "Nexagen Pharmaceuticals",          # must come after the longer form
    ]
    _protected: Dict[str, str] = {}
    _prot_idx = [0]

    for term in PROTECTED_TERMS:
        def _protect_term(m: re.Match, _idx: list = _prot_idx) -> str:
            tok = f'[[prot_{_idx[0]}]]'
            _idx[0] += 1
            _protected[tok] = m.group(0)
            return tok
        redacted = re.sub(re.escape(term), _protect_term, redacted, flags=re.IGNORECASE)

    # ── Step 0a: Protect separator / decoration lines ─────────────────────────
    # Lines consisting entirely of box-drawing or dash characters (e.g. ═══, ───)
    # can be partially consumed by ORG/LOCATION patterns.  Pin them as sentinels
    # before all other processing and restore them at the very end.
    # Tokens include '[' so the NER guard (checks for '[' in ent.text /
    # surrounding) automatically skips them during Pass 2.
    _SEP_RE = re.compile(
        r'[═─━╌┄╍┅╎┆╏┇┉\-=_]{4,}(?:[ \t]*[A-Za-z0-9 \t]*)?',
        re.MULTILINE
    )
    _separators: Dict[str, str] = {}
    def _pin_sep(m: re.Match) -> str:
        tok = f'[[SEP_{len(_separators)}]]'
        _separators[tok] = m.group(0)
        return tok
    redacted = _SEP_RE.sub(_pin_sep, redacted)

    # ── Step 0b: Protect legal terms & case identifiers ───────────────────────
    LEGAL_SHIELDS = [
        # Statutory citations
        r'\d+\s+U\.S\.C\.?(?:\s+§+[\s\d\.]+(?:et\s+seq\.)?)?',
        r'Americans\s+with\s+Disabilities\s+Act(?:\s+of\s+\d{4})?(?:\s+\(["\w]+\))?',
        r'Cal\.\s+Gov\'t\s+Code\s+§+[\s\d\.]+(?:et\s+seq\.)?',
        r'California\s+Fair\s+Employment\s+and\s+Housing\s+Act(?:\s+\(["\w]+\))?',
        r'California\s+Family\s+Rights\s+Act(?:\s+\(["\w]+\))?',
        r'Family\s+and\s+Medical\s+Leave\s+Act(?:\s+\(["\w]+\))?',
        r'Fair\s+Employment\s+and\s+Housing\s+Act(?:\s+\(["\w]+\))?',
        r'(?:the\s+)?(?:ADA|FEHA|CFRA|FMLA)(?=\s|,|\.|$)',
        # EEOC / agency charge numbers — NNN-YYYY-NNNNN format
        # These are case identifiers (public record), not personal PII
        r'\b\d{3}-\d{4}-\d{5}\b',
    ]
    shields: Dict[str, str] = {}
    for i, pattern in enumerate(LEGAL_SHIELDS):
        def replacer(m: re.Match, idx: int = i) -> str:
            token = f'__SHIELD_{idx}_{len(shields)}__'
            shields[token] = m.group(0)
            return token
        redacted = re.sub(pattern, replacer, redacted)

    # ── Pass 1: Regex patterns ────────────────────────────────────────────────
    for pattern, placeholder in REGEX_PATTERNS:
        if categories is not None and placeholder not in categories:
            continue
        for match in re.finditer(pattern, redacted, re.IGNORECASE):
            redaction_log.append({
                "original": match.group(),
                "placeholder": placeholder,
                "method": "regex",
            })
        redacted = re.sub(pattern, placeholder, redacted, flags=re.IGNORECASE)

    # ── Post Pass 1: ZIP code cleanup ─────────────────────────────────────────
    def _replace_zip(m: re.Match) -> str:
        redaction_log.append({
            'original': m.group(2),
            'placeholder': '[ZIP_CODE]',
            'method': 'post-regex',
        })
        return m.group(1) + '[ZIP_CODE]'
    redacted = re.sub(
        r'((?:\[STREET_ADDRESS\]|\[LOCATION\])(?:,?\s+(?:\[LOCATION\]|\[STREET_ADDRESS\]))*,?\s+)'
        r'(\d{5})(?!\d)',
        _replace_zip,
        redacted,
    )

    # ── DOB-context numeric dates ─────────────────────────────────────────────
    # Regex date patterns (MM/DD/YYYY, YYYY-MM-DD) are only redacted when
    # immediately preceded by a PII-indicating birthdate label.  Standalone
    # timeline dates (e.g. "September 12, 2022") are case facts — not PII.
    _DOB_DATE_RE = re.compile(
        r'(?i)((?:DOB|Date\s+of\s+Birth|Born|Birthday)\s*:?\s*)'
        r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})'
    )
    def _replace_dob_date(m: re.Match) -> str:
        redaction_log.append({'original': m.group(2), 'placeholder': '[DATE]', 'method': 'dob-date'})
        return m.group(1) + '[DATE]'
    redacted = _DOB_DATE_RE.sub(_replace_dob_date, redacted)

    # ── Pass 2: spaCy NER ─────────────────────────────────────────────────────
    # Pre-processing: ONLY operations that are strictly length-preserving so
    # that character offsets from the NER output map 1-to-1 back onto `redacted`.
    #
    # • Replace leading '(' with ' ' before a capitalised sequence: 1-to-1 ✓
    # • Title-case consecutive ALL-CAPS words: same byte length ✓
    # • DO NOT collapse multi-spaces: that shortens the string and shifts
    #   every subsequent offset, causing "[DATE]ERMINATION"-style corruption. ✗
    ner_input = re.sub(r'\(([A-Z][a-zA-Z\s]+)', r' \1', redacted)
    def _titlecase_allcaps(m: re.Match) -> str:
        return m.group(0).title()
    ner_input = re.sub(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,}', _titlecase_allcaps, ner_input)

    doc = nlp(ner_input)
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)

    skip_labels = {"CARDINAL", "LAW", "ORDINAL", "PERCENT", "QUANTITY"}
    if aggressive:
        skip_labels = {"ORDINAL", "PERCENT", "QUANTITY"}

    for ent in entities:
        if ent.label_ not in ENTITY_PLACEHOLDERS or ent.label_ in skip_labels:
            continue
        placeholder = ENTITY_PLACEHOLDERS[ent.label_]
        if categories is not None and placeholder not in categories:
            continue

        # ── DATE guards ──────────────────────────────────────────────────────
        if ent.label_ == "DATE":
            txt = ent.text.strip()
            # Guard 1: bare 4-digit year without a preceding month name
            if re.fullmatch(r'\d{4}', txt):
                pre = redacted[max(0, ent.start_char - 20):ent.start_char].strip()
                _MONTHS = ("January","February","March","April","May","June",
                           "July","August","September","October","November","December",
                           "Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
                if not any(pre.endswith(mn) or pre.endswith(mn + ",") for mn in _MONTHS):
                    continue
            # Guard 2: fiscal/generic year references
            if (re.search(r'(?i)\bfiscal\s+year\s+\d{4}\b', txt) or
                    re.fullmatch(r'(?i)year\s+\d{4}', txt)):
                continue
            # Guard 3: entity is purely uppercase letters with no digit — spaCy
            # sometimes mislabels all-caps words like "TERMINATION" as DATE
            if re.fullmatch(r'[A-Z\s]{3,}', txt):
                continue
            # Guard 4: only redact dates in explicit DOB/birthdate context.
            # Timeline dates like "September 12, 2022" are case facts, not PII.
            pre = redacted[max(0, ent.start_char - 80):ent.start_char]
            if not re.search(
                r'(?i)\b(?:DOB|Date\s+of\s+Birth|Born|Birthday)\s*:?\s*$',
                pre
            ):
                continue

        # ── ORG guard — skip job-title phrases ───────────────────────────────
        if ent.label_ == "ORG":
            _JOB_WORDS = {
                "Scientist","Analyst","Engineer","Counsel","Officer","Manager",
                "Director","Consultant","Coordinator","Specialist","Administrator",
                "Advisor","Architect","Developer","Designer","Researcher",
                "Associate","Assistant","Technician","Supervisor",
            }
            _TITLE_PREFIXES = {
                "Senior","Principal","Associate","Chief","Lead","Staff",
                "Junior","Executive","Head","Deputy",
            }
            words = ent.text.split()
            if (len(words) >= 2 and
                    words[0] in _TITLE_PREFIXES and
                    words[-1] in _JOB_WORDS):
                continue

        # Skip anything that already contains a placeholder token
        if "[" in ent.text or "]" in ent.text:
            continue
        surrounding = redacted[max(0, ent.start_char - 1):ent.end_char + 1]
        if "[" in surrounding or "]" in surrounding:
            continue

        redaction_log.append({
            "original": ent.text,
            "placeholder": placeholder,
            "method": f"spacy:{ent.label_}",
        })
        redacted = (
            redacted[:ent.start_char]
            + placeholder
            + redacted[ent.end_char:]
        )

    # ── Restore protected legal terms ─────────────────────────────────────────
    for token, original_text in shields.items():
        redacted = redacted.replace(token, original_text)

    # ── Pass 2.5: City name cleanup ───────────────────────────────────────────
    _CITIES = (
        r'\b(?:San Jose|San Francisco|Los Angeles|San Diego|Sacramento|'
        r'Oakland|Berkeley|Palo Alto|Santa Clara|Sunnyvale|Mountain View|'
        r'Menlo Park|Redwood City|San Mateo|Burlingame|Walnut Creek|'
        r'Fremont|San Ramon|Pleasanton|Livermore|Modesto|Fresno|'
        r'Long Beach|Anaheim|Santa Ana|Riverside|Bakersfield)\b'
    )
    redacted = re.sub(_CITIES, '[LOCATION]', redacted)

    # ── Pass 2.6: Post-NER person-name re-sweep ───────────────────────────────
    # spaCy may miss names in header/form contexts (TO:, FROM:, signature lines,
    # CHARGING PARTY SIGNATURE, etc.).  Collect every multi-word person name
    # already redacted by NER and re-apply them as case-insensitive regex so
    # every remaining occurrence is caught.
    _known_names: List[str] = []
    for entry in redaction_log:
        if entry["placeholder"] == "[PARTY_NAME]":
            name = entry["original"].strip()
            # Ignore single-word hits and placeholder tokens
            if " " in name and "[" not in name and name not in _known_names:
                _known_names.append(name)
                # Also add first+last without middle initial (e.g. "Diana Chen"
                # from "Diana Wei Chen")
                parts = name.split()
                if len(parts) >= 3:
                    first_last = f"{parts[0]} {parts[-1]}"
                    if first_last not in _known_names:
                        _known_names.append(first_last)

    # Add standalone last-name components so surname-only references in
    # timeline tables (e.g. "Chen hired as ...") are caught by the sweep below.
    for name in list(_known_names):
        parts = name.split()
        if len(parts) >= 2:
            last = parts[-1]
            if len(last) >= 3 and last not in _known_names:
                _known_names.append(last)

    for name in _known_names:
        # Use letter-only lookaround instead of \b so names adjacent to pipes,
        # underscores, and other non-letter characters (common in formatted
        # table rows and signature blanks) are still caught.
        pattern = r'(?<![A-Za-z])' + re.escape(name) + r'(?![A-Za-z])'
        hits = re.findall(pattern, redacted, re.IGNORECASE)
        for h in hits:
            redaction_log.append({
                "original": h,
                "placeholder": "[PARTY_NAME]",
                "method": "post-ner-name-sweep",
            })
        redacted = re.sub(pattern, '[PARTY_NAME]', redacted, flags=re.IGNORECASE)

    # Supplementary line-anchored pass: catches names alone on a line
    # (acknowledgment/signature blocks).  Uses \s* so non-breaking spaces,
    # tabs, and carriage returns are all treated as ignorable whitespace.
    for name in _known_names:
        line_pattern = r'^\s*' + re.escape(name) + r'\s*$'
        hits = re.findall(line_pattern, redacted, re.IGNORECASE | re.MULTILINE)
        for h in hits:
            redaction_log.append({
                'original': h.strip(),
                'placeholder': '[PARTY_NAME]',
                'method': 'post-ner-line-sweep',
            })
        redacted = re.sub(line_pattern, '[PARTY_NAME]', redacted,
                          flags=re.IGNORECASE | re.MULTILINE)

    # ── Pass 3: Compound entity merge ─────────────────────────────────────────
    redacted = re.sub(
        r'(?<![\[\w])([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:at|of|for|and)\s+)\[ORGANIZATION\]',
        lambda m: '[ORGANIZATION]',
        redacted,
    )
    redacted = re.sub(
        r'\[ORGANIZATION\](\s+(?:at|of|for|and)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        '[ORGANIZATION]',
        redacted,
    )

    # ── Restore separator lines ────────────────────────────────────────────────
    for token, original_text in _separators.items():
        redacted = redacted.replace(token, original_text)

    # ── Restore protected terms ────────────────────────────────────────────────
    # Must run last — after all passes including compound merge and separator
    # restoration — so protected text is never touched by any redaction logic.
    for token, original_text in _protected.items():
        redacted = redacted.replace(token, original_text)

    placeholder_counts = _count_placeholders(redacted)
    report = {
        "total_redactions": len(redaction_log),
        "redactions": redaction_log,
        "placeholder_counts": placeholder_counts,
        "categories": placeholder_counts,   # alias used by some callers
    }

    return redacted, report

def _count_placeholders(text: str) -> Dict[str, int]:
    """Count occurrences of each placeholder type in redacted text."""
    placeholders = re.findall(r'\[[A-Z_]+\]', text)
    counts = {}
    for p in placeholders:
        counts[p] = counts.get(p, 0) + 1
    return counts

def redact_document(file_path: str, output_path: str = None,
                    aggressive: bool = False,
                    categories: List[str] = None) -> Dict:
    """
    Redact a full document file and save the redacted version.
    Always extracts clean text first via the ingestion pipeline
    before redacting — handles PDF, DOCX, XLSX, TXT, CSV uniformly.
    
    Returns a report dict with redaction statistics.
    """
    file_path = Path(file_path)
    # Detect PDF binary disguised as other formats
    if file_path.suffix.lower() != ".pdf" and is_pdf_binary(str(file_path)):
        return {
            "input_file": str(file_path),
            "output_file": None,
            "status": "error — this file contains PDF binary but has a non-PDF extension. Please rename it to .pdf and re-upload.",
            "total_redactions": 0,
            "redactions": [],
            "placeholder_counts": {}
        }

    # Always use the ingestion extractor for clean text
    # This strips PDF binary, DOCX XML, XLSX markup etc.
    from modules.ingestion import extract_text
    
    try:
        pages = extract_text(str(file_path))
    except ValueError as e:
        return {
            "input_file": str(file_path),
            "output_file": None,
            "status": f"error — {str(e)}",
            "total_redactions": 0,
            "redactions": [],
            "placeholder_counts": {}
        }

    if not pages:
        return {
            "input_file": str(file_path),
            "output_file": None,
            "status": "error — no extractable text (possibly image-based PDF)",
            "total_redactions": 0,
            "redactions": [],
            "placeholder_counts": {}
        }

    # Join all pages with clear page separators
    text = "\n\n--- PAGE BREAK ---\n\n".join([p["text"] for p in pages])

    # Redact
    redacted_text, report = redact_text(text, aggressive=aggressive,
                                        categories=categories)

    # Save output
    if output_path is None:
        output_path = file_path.parent / f"{file_path.stem}_REDACTED.txt"

    with open(str(output_path), "w", encoding="utf-8") as f:
        f.write(redacted_text)

    report["input_file"] = str(file_path)
    report["output_file"] = str(output_path)
    report["status"] = "success"

    return report

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) > 1:
        path = sys.argv[1]
        aggressive = "--aggressive" in sys.argv

        print(f"\nRedacting: {path}")
        print(f"Mode: {'aggressive' if aggressive else 'standard'}")
        print("=" * 60)

        report = redact_document(path, aggressive=aggressive)

        print(f"\n✅ Done. Redacted file saved to: {report['output_file']}")
        print(f"\nRedaction Summary:")
        print(f"  Total redactions: {report['total_redactions']}")
        print(f"\n  Placeholder counts:")
        for placeholder, count in report['placeholder_counts'].items():
            print(f"    {placeholder}: {count}")

        print(f"\n  Sample redactions (first 10):")
        for r in report['redactions'][:10]:
            print(f"    '{r['original']}' → {r['placeholder']} ({r['method']})")
    else:
        print("Usage: python -m modules.redaction <file_path> [--aggressive]")
        print("\nRunning quick test on sample text...")

        sample = """
        My client John Smith (SSN: 123-45-6789) is filing a complaint against 
        Acme Corporation on January 15, 2024. The case number is 24-CV-00123.
        Mr. Smith can be reached at john.smith@email.com or (555) 123-4567.
        He is seeking damages of $250,000 from the defendant based in New York.
        Attorney Mary Johnson, Bar No. 98765, represents the plaintiff.
        """

        print("\nOriginal text:")
        print(sample)

        redacted, report = redact_text(sample)

        print("\nRedacted text:")
        print(redacted)

        print(f"\nTotal redactions: {report['total_redactions']}")
        for placeholder, count in report['placeholder_counts'].items():
            print(f"  {placeholder}: {count}")