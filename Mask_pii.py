"""
EDD Narrative PII Masker — regex-based
=======================================
- Reads a raw CSV, masks PII in the narrative column, writes a masked CSV.
- Configurable: toggle which categories of PII to mask via ENABLED_CATEGORIES.
- Reversible: writes a lookup JSON so you can unmask model outputs later.
- Same value within one narrative gets the same placeholder (e.g. "John Smith"
  appearing 3 times becomes [PERSON_1] all 3 times, not PERSON_1/2/3).

Usage:
    1. Edit the CONFIG block below (paths + which categories to mask).
    2. Run:  python mask_pii.py
    3. Point your classification pipeline at OUTPUT_CSV_PATH.
    4. To unmask model outputs later, use unmask_text() with the saved lookup.

IMPORTANT LIMITATIONS OF REGEX-BASED MASKING:
- Cannot reliably mask person names, company names, or free-form addresses.
  For those you need an NER approach (spaCy, Presidio). Names pattern below
  catches only "Mr. X" / "Ms. Y" / obvious title+name forms as a weak best-effort.
- Will occasionally over-mask: a ZIP code pattern can catch any 5-digit number;
  a DATE pattern can catch things that look like dates but aren't.
- Will occasionally under-mask: unconventional formatting slips past regex.
- Order matters: more specific patterns run before generic ones so they claim
  their matches first (SSN before generic 9-digit, credit card before account).
"""

import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ==============================================================
# CONFIG
# ==============================================================

BASE_DIR = "/Users/bodhit/GenAi - Narrative Classification"

INPUT_CSV_PATH   = f"{BASE_DIR}/edd_final.csv"
OUTPUT_CSV_PATH  = f"{BASE_DIR}/edd_final_masked.csv"
LOOKUP_JSON_PATH = f"{BASE_DIR}/masking_lookup.json"  # keyed by row index; used for unmasking

NARRATIVE_COLUMN = "narrative"

# Which categories to mask. Comment out any you want to leave unmasked.
# Order in this list also determines match priority (earlier = claims matches first).
ENABLED_CATEGORIES = [
    "EMAIL",          # most specific — run first
    "URL",
    "IP_ADDRESS",
    "SSN",
    "CREDIT_CARD",
    "IBAN",
    "SWIFT_BIC",
    "ACCOUNT_NUMBER", # context-anchored ("account <N>"), run BEFORE phone
    "PASSPORT",       # context-anchored ("passport <X>"), run BEFORE phone
    "PHONE",
    "MONEY",
    "DATE",
    "ZIP_CODE",
    "PERSON_TITLE",   # weak — only catches "Mr./Ms./Mrs./Dr. Name" patterns
]


# ==============================================================
# PATTERN LIBRARY
# ==============================================================
# Each pattern is compiled once up top for speed. Add/edit patterns here.

PATTERNS = {
    # Contact info
    "EMAIL":         re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "URL":           re.compile(r"https?://[^\s,\)]+", re.IGNORECASE),
    "IP_ADDRESS":    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "PHONE":         re.compile(
        # Requires at least one separator between digit groups so plain 10-digit
        # account numbers don't match. Uses digit-boundary lookarounds instead of
        # \b so parens and + prefixes anchor correctly.
        r"(?<!\d)(?:\+\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
    ),

    # Government / financial identifiers
    "SSN":           re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD":   re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "IBAN":          re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "SWIFT_BIC":     re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
    "ACCOUNT_NUMBER": re.compile(
        r"\b(?:a\/c|acct?|account)(?:\s*(?:no\.?|number|#))?[\s:#]*([A-Z0-9\-]{6,20})\b",
        re.IGNORECASE,
    ),
    "PASSPORT":      re.compile(
        r"\b(?:passport|pp)(?:\s*(?:no\.?|number|#))?[\s:#]*([A-Z0-9]{6,12})\b",
        re.IGNORECASE,
    ),

    # Monetary amounts — currency symbol OR amount followed by currency code
    "MONEY":         re.compile(
        r"(?:[\$€£¥₹]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
        r"|"
        r"(?:\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(?:USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|SGD|HKD)\b)"
    ),

    # Dates — several common formats
    "DATE":          re.compile(
        r"\b(?:\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})"          # 12/31/2024, 31-12-24
        r"|"
        r"(?:\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})"              # 2024-12-31
        r"|"
        r"(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})"  # 15 Mar 2024
        r"|"
        r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})"  # Mar 15, 2024
        r"\b",
        re.IGNORECASE,
    ),

    # US ZIP — last because it's greedy (any 5-digit number)
    "ZIP_CODE":      re.compile(r"\b\d{5}(?:-\d{4})?\b"),

    # Weak name catcher — only "Mr./Ms./Mrs./Dr./Mx. Firstname [Lastname]"
    "PERSON_TITLE":  re.compile(
        r"\b(?:Mr|Mrs|Ms|Mx|Dr|Miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
    ),
}


# ==============================================================
# CORE MASKING LOGIC
# ==============================================================

def mask_text(text: str, categories: list) -> tuple[str, dict]:
    """
    Mask PII in a single string based on `categories` (order-sensitive).
    Returns (masked_text, lookup) where lookup maps placeholder -> original value.
    Identical values within the same text share the same placeholder.
    """
    if pd.isna(text) or not str(text).strip():
        return text, {}

    working = str(text)
    lookup: dict[str, str] = {}
    counter: dict[str, int] = defaultdict(int)
    value_to_placeholder: dict[str, str] = {}  # deduplicate within this narrative

    for cat in categories:
        pat = PATTERNS.get(cat)
        if pat is None:
            continue

        def _replace(match, cat=cat):
            value = match.group(0)
            # Reuse placeholder if this exact value was already seen in this narrative
            if value in value_to_placeholder:
                return value_to_placeholder[value]
            counter[cat] += 1
            placeholder = f"[{cat}_{counter[cat]}]"
            value_to_placeholder[value] = placeholder
            lookup[placeholder] = value
            return placeholder

        working = pat.sub(_replace, working)

    return working, lookup


def mask_dataframe(
    df: pd.DataFrame,
    narrative_col: str = NARRATIVE_COLUMN,
    categories: list = None,
    id_col: str = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Mask narratives across a DataFrame.
    Returns (masked_df, full_lookup) where full_lookup is keyed by row identifier.
    If id_col is None, uses the DataFrame's positional index as the key.
    """
    if categories is None:
        categories = ENABLED_CATEGORIES

    masked_col = []
    full_lookup: dict[str, dict] = {}
    stats: dict[str, int] = defaultdict(int)

    for pos, (_, row) in enumerate(df.iterrows()):
        row_key = str(row[id_col]) if id_col and id_col in df.columns else str(pos)
        masked, lookup = mask_text(row[narrative_col], categories)
        masked_col.append(masked)
        if lookup:
            full_lookup[row_key] = lookup
            for placeholder in lookup:
                cat = placeholder.strip("[]").rsplit("_", 1)[0]
                stats[cat] += 1

    result = df.copy()
    result[narrative_col] = masked_col

    print("\nMasking stats (total replacements across all rows):")
    if not stats:
        print("  (nothing masked — check your patterns and input data)")
    for cat in categories:
        if stats.get(cat, 0):
            print(f"  {cat:<16} {stats[cat]:,}")

    rows_with_any_mask = len(full_lookup)
    print(f"\nRows with at least one mask applied: {rows_with_any_mask:,} / {len(df):,}")

    return result, full_lookup


def unmask_text(masked_text: str, lookup: dict) -> str:
    """Reverse masking on a single string using its lookup dict."""
    result = masked_text
    # Replace longer placeholders first so [X_10] doesn't get mangled by [X_1]
    for placeholder, original in sorted(lookup.items(), key=lambda kv: -len(kv[0])):
        result = result.replace(placeholder, original)
    return result


# ==============================================================
# MAIN
# ==============================================================

def main():
    print(f"Loading {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Loaded {len(df):,} rows")

    if NARRATIVE_COLUMN not in df.columns:
        raise KeyError(
            f"Column '{NARRATIVE_COLUMN}' not found. Available: {list(df.columns)}"
        )

    print(f"\nMasking categories: {ENABLED_CATEGORIES}")

    masked_df, lookup = mask_dataframe(df, NARRATIVE_COLUMN, ENABLED_CATEGORIES)

    Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
    masked_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nWrote masked CSV -> {OUTPUT_CSV_PATH}")

    with open(LOOKUP_JSON_PATH, "w") as f:
        json.dump(lookup, f, indent=2, ensure_ascii=False)
    print(f"Wrote lookup JSON -> {LOOKUP_JSON_PATH}")

    # Show a before/after preview of the first row that actually had PII
    preview_row = None
    for i in range(len(df)):
        if str(i) in lookup:
            preview_row = i
            break

    if preview_row is not None:
        print(f"\n--- Preview (row index {preview_row}) ---")
        print("BEFORE:")
        print(f"  {df.iloc[preview_row][NARRATIVE_COLUMN][:400]}")
        print("AFTER:")
        print(f"  {masked_df.iloc[preview_row][NARRATIVE_COLUMN][:400]}")


if __name__ == "__main__":
    main()