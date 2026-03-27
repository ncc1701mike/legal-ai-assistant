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
    "MONEY":      "[FINANCIAL_AMOUNT]",
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
    # Dollar amounts — minimum 2 digits to avoid PDF artifacts
    (r'\$[\d,]{2,}(\.\d{2})?',                                    "[FINANCIAL_AMOUNT]"),
    # International phone numbers with spaces — +44 20 7946 0958
    (r'\+\d{1,3}[\s.-]\d{1,4}[\s.-]\d{3,4}[\s.-]\d{3,4}\b', '[PHONE]'),
     # Bar numbers — must come before case numbers
    (r'\bBar\s+No\.?\s*\d+',                                       "[BAR_NO]"),
    # Bates numbers — alphanumeric with hyphens or underscores
    (r'\b[A-Z]{2,}[_-][A-Z0-9]{2,}[_-]\d{4,}\b', '[BATES_NO]'),
    # Docket numbers
    (r'\bDocket\s+No\.?\s*[\w\-]+',                                "[DOCKET_NO]"),
    # Case numbers — specific formats
    (r'\bCase\s+No\.?\s*[\w\-]+',                                  "[CASE_NO]"),
    (r'\bNo\.\s*\d[\w\-]*',                                        "[CASE_NO]"),
    # Alphanumeric case numbers like 24-CV-00123
    (r'\b\d{2,4}-[A-Z]{2,}-\d{4,}\b',                             "[CASE_NO]"),
    # Pure numeric case numbers like 123-456
    (r'\b\d{1,3}-\d{3,6}\b',                                       "[CASE_NO]"),
    # Dates in MM/DD/YYYY and YYYY-MM-DD formats
    (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]'),
    (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),
    # Street addresses — number + street name + street type
    (r'\b\d{3,5}\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl|Way|Circle|Cir|Trail|Tr|Terrace|Ter|Highway|Hwy)(?:\.|\b)(?:,?\s+(?:Suite|Ste|Apt|Unit|Floor|Fl)\.?\s*[\w]+)?',
     '[STREET_ADDRESS]'),
    # ZIP codes — only match in address context (after comma+space or space after state)
    (r'(?<=,\s)\d{5}(?!\d)',                                       "[ZIP_CODE]"),
    (r'(?<=\s)[A-Z]{2}\s\d{5}(?!\d)',                             "[ZIP_CODE]"),
    # ZIP after already-redacted location placeholder — handled in post-processing
    # Leading parenthesis before capitalized name/org — (Wayland Station...
    (r'\(([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)', '[ORGANIZATION]'),
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

    # ── Protect legal terms from over-redaction ──────────────────────────────
    LEGAL_SHIELDS = [
        r'\d+\s+U\.S\.C\.?(?:\s+§+[\s\d\.]+(?:et\s+seq\.)?)?',
        r'Americans\s+with\s+Disabilities\s+Act(?:\s+of\s+\d{4})?(?:\s+\(["\w]+\))?',
        r'Cal\.\s+Gov\'t\s+Code\s+§+[\s\d\.]+(?:et\s+seq\.)?',
        r'California\s+Fair\s+Employment\s+and\s+Housing\s+Act(?:\s+\(["\w]+\))?',
        r'California\s+Family\s+Rights\s+Act(?:\s+\(["\w]+\))?',
        r'Family\s+and\s+Medical\s+Leave\s+Act(?:\s+\(["\w]+\))?',
        r'Fair\s+Employment\s+and\s+Housing\s+Act(?:\s+\(["\w]+\))?',
        r'(?:the\s+)?(?:ADA|FEHA|CFRA|FMLA)(?=\s|,|\.|$)',
    ]
    shields = {}
    for i, pattern in enumerate(LEGAL_SHIELDS):
        def replacer(m, idx=i):
            token = f'__SHIELD_{idx}_{len(shields)}__'
            shields[token] = m.group(0)
            return token
        redacted = re.sub(pattern, replacer, redacted)


    # ── Pass 1: Regex patterns ────────────────────────────────────────────────
    for pattern, placeholder in REGEX_PATTERNS:
        # Skip if category filtering is active and this placeholder not selected
        if categories is not None and placeholder not in categories:
            continue
        matches = re.finditer(pattern, redacted, re.IGNORECASE)
        for match in matches:
            original = match.group()
            redaction_log.append({
                "original": original,
                "placeholder": placeholder,
                "method": "regex"
            })
        redacted = re.sub(pattern, placeholder, redacted, flags=re.IGNORECASE)

    # ── Post Pass 1: ZIP code cleanup ──────────────────────────────────────────
    # Catch ZIPs that follow any combination of [LOCATION]/[STREET_ADDRESS] placeholders
    import re as _re
    def _replace_zip(m):
        prefix = m.group(1)
        zip_code = m.group(2)
        redaction_log.append({'original': zip_code, 'placeholder': '[ZIP_CODE]', 'method': 'post-regex'})
        return prefix + '[ZIP_CODE]'
    redacted = _re.sub(
        r'((?:\[STREET_ADDRESS\]|\[LOCATION\])(?:,?\s+(?:\[LOCATION\]|\[STREET_ADDRESS\]))*,?\s+)(\d{5})(?!\d)',
        _replace_zip,
        redacted
    )

    # ── Pass 2: spaCy NER ─────────────────────────────────────────────────────
    # Pre-processing: normalize punctuation that confuses NER boundaries
    # Fix 1: leading parenthesis before entity — (Wayland Station...
    ner_input = re.sub(r'\(([A-Z][a-zA-Z\s]+)', r'\1', redacted)
    # Fix 2: title-case all-caps sequences so spaCy ORG tagger fires
    def _titlecase_allcaps(m):
        return m.group(0).title()
    ner_input = re.sub(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,}){1,}', _titlecase_allcaps, ner_input)
    doc = nlp(ner_input)
    
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    skip_labels = {"CARDINAL", "LAW", "ORDINAL", "PERCENT", "QUANTITY"}
    if aggressive:
        skip_labels = {"ORDINAL", "PERCENT", "QUANTITY"}

    for ent in entities:
        if ent.label_ in ENTITY_PLACEHOLDERS and ent.label_ not in skip_labels:
            placeholder = ENTITY_PLACEHOLDERS[ent.label_]
            # Skip if category filtering is active and this placeholder not selected
            if categories is not None and placeholder not in categories:
                continue
            # Skip bare 4-digit years and fiscal year references
            if ent.label_ == "DATE":
                txt = ent.text.strip()
                # Case 1: bare year only e.g. "2022"
                if re.fullmatch(r'\d{4}', txt):
                    pre = redacted[max(0, ent.start_char-20):ent.start_char].strip()
                    month_names = ("January","February","March","April","May","June",
                                   "July","August","September","October","November","December",
                                   "Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Oct","Nov","Dec")
                    if not any(pre.endswith(m) or pre.endswith(m+",") for m in month_names):
                        continue
                # Case 2: "fiscal year 2022" or "year 2022" — not a real date
                if re.search(r'(?i)\bfiscal\s+year\s+\d{4}\b', txt) or \
                   re.fullmatch(r'(?i)year\s+\d{4}', txt):
                    continue
            # Skip job titles misclassified as ORG by spaCy
            if ent.label_ == "ORG":
                JOB_TITLE_WORDS = {
                    "Scientist", "Analyst", "Engineer", "Counsel", "Officer",
                    "Manager", "Director", "Consultant", "Coordinator", "Specialist",
                    "Administrator", "Advisor", "Architect", "Developer", "Designer",
                    "Researcher", "Associate", "Assistant", "Technician", "Supervisor"
                }
                title_prefixes = {
                    "Senior", "Principal", "Associate", "Chief", "Lead", "Staff",
                    "Junior", "Executive", "Head", "Deputy"
                }
                words = ent.text.split()
                if (len(words) >= 2 and
                    words[0] in title_prefixes and
                    words[-1] in JOB_TITLE_WORDS):
                    continue
            # Skip anything already redacted
            if "[" in ent.text or "]" in ent.text:
                continue
            surrounding = redacted[max(0, ent.start_char-1):ent.end_char+1]
            if "[" in surrounding or "]" in surrounding:
                continue
            redaction_log.append({
                "original": ent.text,
                "placeholder": placeholder,
                "method": f"spacy:{ent.label_}"
            })
            redacted = (
                redacted[:ent.start_char]
                + placeholder
                + redacted[ent.end_char:]
            )

    # ── Restore protected legal terms ─────────────────────────────────────────
    for token, original in shields.items():
        redacted = redacted.replace(token, original)


    # ── Pass 3: Compound entity merge ────────────────────────────────────────
    # Fix: partial org name before/after [ORGANIZATION] placeholder
    redacted = re.sub(
        r'(?<![\[\w])([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:at|of|for|and)\s+)\[ORGANIZATION\]',
        lambda m: '[ORGANIZATION]',
        redacted
    )
    redacted = re.sub(
        r'\[ORGANIZATION\](\s+(?:at|of|for|and)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        '[ORGANIZATION]',
        redacted
    )
    report = {
        "total_redactions": len(redaction_log),
        "redactions": redaction_log,
        "placeholder_counts": _count_placeholders(redacted)
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