"""
EDD (Enhanced Due Diligence) Narrative Classification Pipeline
================================================================
- Stratified sampling on `Decision` column (proportions preserved)
- Assigns independent row_number (1..N) as the only join key
- PII-safe: only [row_number, narrative] ever reach the model
- Two approaches: local Ollama (llama3.1:8b) OR Claude API
- Output: 4-column CSV (row_number, label, confidence_score, reason)

Run:
    python edd_classification_pipeline.py
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ==============================================================
# CONFIGURATION — edit these constants
# ==============================================================

# --- File paths ---
# Single source of truth for where everything lives. Change this one line to relocate
# the whole project. All input, output, checkpoint, and comparison files are derived
# from this folder.
BASE_DIR = "/Users/bodhit/GenAi - Narrative Classification"

# Input
INPUT_CSV_PATH   = f"{BASE_DIR}/edd_final.csv"

# Sampled rows kept with row_number for offline join later (narrative not passed to model)
SAMPLED_CSV_PATH = f"{BASE_DIR}/edd_sampled_with_row_number.csv"

# Each approach writes to its OWN output file so runs don't overwrite each other.
# Run with APPROACH="ollama" once, then APPROACH="claude" once — both files persist side-by-side.
OUTPUT_CSV_PATH_OLLAMA     = f"{BASE_DIR}/edd_classifications_ollama.csv"
OUTPUT_CSV_PATH_CLAUDE     = f"{BASE_DIR}/edd_classifications_claude.csv"
CHECKPOINT_CSV_PATH_CLAUDE = f"{BASE_DIR}/edd_classifications_claude.checkpoint.csv"

# Side-by-side comparison output (see compare_outputs() at bottom)
COMPARISON_CSV_PATH = f"{BASE_DIR}/edd_comparison_ollama_vs_claude.csv"

# --- Column names in source CSV ---
DECISION_COLUMN  = "Decision"
NARRATIVE_COLUMN = "narrative"

# --- Sampling ---
SAMPLE_SIZE  = 1000
RANDOM_SEED  = 42

# --- Approach selection: "ollama" or "claude" ---
APPROACH = "claude"

# --- Ollama (local) ---
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT_SEC = 120

# --- Claude API ---
CLAUDE_MODEL         = "claude-sonnet-4-6"

# Path to a plain-text file containing ONLY your Claude API key (no quotes, no "export",
# nothing else). The script strips whitespace/newlines automatically. If this file is
# missing or empty, the script falls back to the ANTHROPIC_API_KEY environment variable.
CLAUDE_API_KEY_FILE = f"{BASE_DIR}/api_key.txt"


def _load_claude_api_key(file_path: str) -> str:
    """Prefer key from file; fall back to ANTHROPIC_API_KEY env var."""
    try:
        p = Path(file_path)
        if p.exists():
            key = p.read_text().strip()
            if key:
                return key
    except Exception as e:
        print(f"[warn] Could not read {file_path}: {e}")
    return os.environ.get("ANTHROPIC_API_KEY", "")


CLAUDE_API_KEY       = _load_claude_api_key(CLAUDE_API_KEY_FILE)
CHECKPOINT_INTERVAL  = 500
MAX_RETRIES          = 5
INITIAL_BACKOFF_SEC  = 2
CLAUDE_MAX_TOKENS    = 500

# ==============================================================
# PROMPTS (shared by both approaches)
# ==============================================================

SYSTEM_PROMPT = """ROLE
You are a senior AML analyst assistant. Your task is to read Enhanced Due Diligence (EDD) narratives and classify the outcome of each review.

TASK
You will receive a single EDD narrative. For that narrative you must:
- Determine edd_outcome
- Assign a confidence_score
- Provide a confidence_reason

OUTPUT FORMAT (STRICT)
Return ONLY a JSON object with exactly these three keys. No preamble, no markdown fences, no commentary, nothing before or after the JSON:
{
  "edd_outcome": 0 or 1,
  "confidence_score": 0.00 to 1.00,
  "confidence_reason": "short sentence explaining your certainty"
}

EDD OUTCOME RULES

Precedence: Always evaluate the edd_outcome = 1 conditions first. If ANY of them is present, assign 1. Only if none of the 1-conditions are present should you evaluate the 0-conditions. Adverse action always wins over clean-case indicators.

edd_outcome = 1 (POSITIVE — something was found)
Assign 1 if ANY of the following is present:
- SAR filed or recommended
- Account closed, restricted, or cancelled
- Relationship exited or terminated
- Law enforcement involvement (e.g., FinCEN referral, legal hold)
- Confirmed financial crime indicators
- Source of wealth unverifiable AND escalation/action taken
- Explicit mention of fraud, laundering, structuring, sanctions evasion, or misuse confirmed by investigation

edd_outcome = 0 (NEGATIVE — clean review)
Assign 0 if ANY of the following is true (and none of the 1-conditions above apply):
- No suspicious activity identified
- No SAR, referral, or legal hold
- Account remains open
- Outcome states: case closed / no action / monitoring only
- Account scheduled for another review

Important rule: "Enhanced monitoring recommended" alone = 0

CONFIDENCE SCORE GUIDELINES
- 0.90 – 1.00 → Explicit outcome (e.g., SAR filed, account closed)
- 0.75 – 0.89 → Strong signal, slight interpretation needed
- 0.60 – 0.74 → Likely but somewhat unclear
- 0.50 – 0.59 → Borderline / vague
- < 0.50 → Unclear → needs human review

CONFIDENCE REASON
- One short sentence explaining your certainty
- Focus on the key signal or the key ambiguity
Examples:
- "SAR filed and account closed — explicit outcome."
- "Only monitoring recommended, no adverse action."
- "Narrative unclear and lacks final disposition."

FINAL INSTRUCTION
Return ONLY the JSON object. No other text."""

USER_PROMPT_TEMPLATE = "Narrative:\n\n{narrative}"


# ==============================================================
# STEP 1 — Load CSV + stratified sample on Decision
# ==============================================================

def stratified_sample(df: pd.DataFrame, col: str, n: int, seed: int) -> pd.DataFrame:
    """
    Proportional allocation: for each unique value in `col`, sample
    round(proportion * n) rows. Preserves class proportions as closely
    as integer row counts allow.
    """
    counts = df[col].value_counts()
    total  = counts.sum()

    pieces = []
    for label, cnt in counts.items():
        prop = cnt / total
        k    = max(1, round(prop * n))
        k    = min(k, cnt)                  # never oversample
        grp  = df[df[col] == label]
        pieces.append(grp.sample(n=k, random_state=seed))

    sampled = pd.concat(pieces, ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled


def load_and_sample(path: str, decision_col: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    if NARRATIVE_COLUMN not in df.columns:
        raise KeyError(f"Column '{NARRATIVE_COLUMN}' not found. Available: {list(df.columns)}")

    # Show Decision distribution for reference (even though we're not stratifying on it right now)
    if decision_col in df.columns:
        orig = df[decision_col].value_counts(normalize=True)
        print("\nOriginal Decision distribution (informational only — not used for sampling):")
        for k, v in orig.items():
            print(f"  {k!s:<40} {v*100:6.2f}%")

    # ---- TEMPORARY: simple random sampling on rows with a valid narrative ----
    # Drop rows where narrative is null or empty/whitespace-only
    valid = df[df[NARRATIVE_COLUMN].notna() & df[NARRATIVE_COLUMN].astype(str).str.strip().ne("")]
    dropped = len(df) - len(valid)
    if dropped:
        print(f"\nExcluded {dropped:,} rows with empty/null narrative. {len(valid):,} rows eligible for sampling.")

    k = min(n, len(valid))
    if k < n:
        print(f"Note: requested {n:,} but only {len(valid):,} valid rows available. Sampling {k:,}.")

    sampled = valid.sample(n=k, random_state=seed).reset_index(drop=True)

    # ---- ORIGINAL STRATIFIED SAMPLING (commented out — uncomment to restore) ----
    # sampled = stratified_sample(df, decision_col, n, seed)

    print(f"\nSampled {len(sampled):,} rows (target {n:,})")
    if decision_col in sampled.columns:
        samp = sampled[decision_col].value_counts(normalize=True)
        print("Sampled Decision distribution (unstratified — will differ from original):")
        for k_, v in samp.items():
            print(f"  {k_!s:<40} {v*100:6.2f}%")

    return sampled


# ==============================================================
# STEP 2 — Assign row_number + extract only the model input
# ==============================================================

def prepare_model_input(sampled_df: pd.DataFrame, narrative_col: str):
    """
    Returns (narrative_df, sampled_with_row_number).
      narrative_df              -> 2 cols: row_number, narrative   (goes to model)
      sampled_with_row_number   -> full sampled data + row_number  (kept by caller)
    """
    if narrative_col not in sampled_df.columns:
        raise KeyError(f"Column '{narrative_col}' not found. Available: {list(sampled_df.columns)}")

    sampled = sampled_df.copy()
    sampled["row_number"] = range(1, len(sampled) + 1)

    narrative_df = sampled[["row_number", narrative_col]].rename(
        columns={narrative_col: "narrative"}
    ).reset_index(drop=True)

    return narrative_df, sampled


# ==============================================================
# STEP 3 — Shared JSON parsing helper
# ==============================================================

def parse_model_json(text: str) -> dict:
    """Robust JSON parsing: strips ```json fences if the model added them."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    return json.loads(t)


def normalize_result(row_num: int, parsed: dict) -> dict:
    """Validate + coerce the parsed JSON into the 4-column schema."""
    return {
        "row_number":        row_num,
        "edd_outcome":       int(parsed["edd_outcome"]),
        "confidence_score":  float(parsed["confidence_score"]),
        "confidence_reason": str(parsed["confidence_reason"]),
    }


def error_result(row_num: int, err: Exception | str) -> dict:
    msg = f"ERROR: {type(err).__name__}: {err}" if isinstance(err, Exception) else f"ERROR: {err}"
    return {
        "row_number":        row_num,
        "edd_outcome":       None,
        "confidence_score":  None,
        "confidence_reason": msg,
    }


# ==============================================================
# STEP 3A — Ollama (local llama3.1:8b)
# ==============================================================

def classify_with_ollama(narrative_df: pd.DataFrame) -> pd.DataFrame:
    import requests

    results = []
    for _, row in tqdm(narrative_df.iterrows(), total=len(narrative_df), desc="Ollama"):
        row_num   = int(row["row_number"])
        narrative = row["narrative"]

        try:
            if pd.isna(narrative) or not str(narrative).strip():
                raise ValueError("Empty narrative")

            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(narrative=narrative)},
                ],
                "format": "json",
                "stream": False,
            }
            r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT_SEC)
            r.raise_for_status()
            content = r.json()["message"]["content"]
            parsed  = parse_model_json(content)
            results.append(normalize_result(row_num, parsed))

        except Exception as e:
            tqdm.write(f"[Row {row_num}] {type(e).__name__}: {e}")
            results.append(error_result(row_num, e))

    return pd.DataFrame(results, columns=["row_number", "edd_outcome", "confidence_score", "confidence_reason"])


# ==============================================================
# STEP 3B — Claude API with checkpointing + exponential backoff
# ==============================================================

def classify_with_claude(narrative_df: pd.DataFrame) -> pd.DataFrame:
    from anthropic import Anthropic
    try:
        from anthropic import RateLimitError, APIStatusError, APIConnectionError
    except ImportError:
        # Older anthropic SDK fallback
        from anthropic import RateLimitError, APIError as APIStatusError
        APIConnectionError = APIStatusError

    if not CLAUDE_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (env var or CLAUDE_API_KEY constant).")

    client = Anthropic(api_key=CLAUDE_API_KEY)

    # ---- Resume from checkpoint if present ----
    completed_rows: set[int] = set()
    results: list[dict] = []
    if Path(CHECKPOINT_CSV_PATH_CLAUDE).exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV_PATH_CLAUDE)
        completed_rows = set(int(x) for x in ckpt["row_number"].tolist())
        results        = ckpt.to_dict("records")
        print(f"Resuming from checkpoint: {len(completed_rows):,} rows already done")

    pending = narrative_df[~narrative_df["row_number"].isin(completed_rows)].reset_index(drop=True)
    print(f"Classifying {len(pending):,} remaining rows")

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), desc="Claude")):
        row_num   = int(row["row_number"])
        narrative = row["narrative"]

        parsed_or_err = None

        # ---- Retry loop with exponential backoff ----
        for attempt in range(MAX_RETRIES):
            try:
                if pd.isna(narrative) or not str(narrative).strip():
                    raise ValueError("Empty narrative")

                resp = client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=CLAUDE_MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(narrative=narrative)}
                    ],
                )
                text   = resp.content[0].text
                parsed = parse_model_json(text)
                parsed_or_err = ("ok", parsed)
                break

            except RateLimitError:
                wait = INITIAL_BACKOFF_SEC * (2 ** attempt)
                tqdm.write(f"[Row {row_num}] Rate limited (attempt {attempt+1}/{MAX_RETRIES}). Sleeping {wait}s")
                time.sleep(wait)

            except (APIStatusError, APIConnectionError) as e:
                wait = INITIAL_BACKOFF_SEC * (2 ** attempt)
                tqdm.write(f"[Row {row_num}] API error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Sleeping {wait}s")
                time.sleep(wait)

            except Exception as e:
                # Non-retryable (bad JSON, empty narrative, etc.)
                parsed_or_err = ("err", e)
                break
        else:
            parsed_or_err = ("err", "max retries exceeded")

        # ---- Store result ----
        try:
            status, payload = parsed_or_err
            if status == "ok":
                results.append(normalize_result(row_num, payload))
            else:
                results.append(error_result(row_num, payload))
        except Exception as e:
            results.append(error_result(row_num, e))

        # ---- Checkpoint ----
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_CSV_PATH_CLAUDE, index=False)
            tqdm.write(f"[Checkpoint] Wrote {len(results):,} rows to {CHECKPOINT_CSV_PATH_CLAUDE}")

    return pd.DataFrame(results, columns=["row_number", "edd_outcome", "confidence_score", "confidence_reason"])


# ==============================================================
# MAIN
# ==============================================================

def main():
    Path(SAMPLED_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Step 1: load + stratified sample
    sampled = load_and_sample(INPUT_CSV_PATH, DECISION_COLUMN, SAMPLE_SIZE, RANDOM_SEED)

    # Step 2: assign row_number, extract narrative-only frame
    narrative_df, sampled_with_rn = prepare_model_input(sampled, NARRATIVE_COLUMN)

    # Save the sampled rows WITH row_number so you can join back offline later
    sampled_with_rn.to_csv(SAMPLED_CSV_PATH, index=False)
    print(f"\nSaved sampled data + row_number -> {SAMPLED_CSV_PATH}")
    print(f"Model will see ONLY these {len(narrative_df.columns)} columns: {list(narrative_df.columns)}")

    # Step 3: classify — and pick the matching output path
    if APPROACH == "ollama":
        results_df  = classify_with_ollama(narrative_df)
        output_path = OUTPUT_CSV_PATH_OLLAMA
    elif APPROACH == "claude":
        results_df  = classify_with_claude(narrative_df)
        output_path = OUTPUT_CSV_PATH_CLAUDE
    else:
        raise ValueError(f"Unknown APPROACH: {APPROACH!r}. Use 'ollama' or 'claude'.")

    # Step 4: save 4-col output to the approach-specific path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    n_err = results_df["edd_outcome"].isna().sum()
    print(f"\nDone. Classified {len(results_df):,} rows ({n_err:,} errors) -> {output_path}")

    # Cleanup checkpoint on successful completion
    if APPROACH == "claude" and Path(CHECKPOINT_CSV_PATH_CLAUDE).exists():
        Path(CHECKPOINT_CSV_PATH_CLAUDE).unlink()
        print(f"Removed checkpoint file {CHECKPOINT_CSV_PATH_CLAUDE}")


if __name__ == "__main__":
    main()


# ==============================================================
# COMPARISON UTILITY
# --------------------------------------------------------------
# Once you've run BOTH approaches (APPROACH="ollama" then "claude"),
# run this from a Python shell / notebook:
#
#     from edd_classification_pipeline import compare_outputs
#     compare_outputs()
#
# It joins both CSVs on row_number, shows agreement stats, and
# writes a side-by-side file to COMPARISON_CSV_PATH.
# ==============================================================

def compare_outputs(
    ollama_path: str = OUTPUT_CSV_PATH_OLLAMA,
    claude_path: str = OUTPUT_CSV_PATH_CLAUDE,
    out_path: str = COMPARISON_CSV_PATH,
) -> pd.DataFrame:
    if not Path(ollama_path).exists():
        raise FileNotFoundError(f"Ollama output not found: {ollama_path}")
    if not Path(claude_path).exists():
        raise FileNotFoundError(f"Claude output not found: {claude_path}")

    o = pd.read_csv(ollama_path).add_suffix("_ollama").rename(columns={"row_number_ollama": "row_number"})
    c = pd.read_csv(claude_path).add_suffix("_claude").rename(columns={"row_number_claude": "row_number"})

    merged = o.merge(c, on="row_number", how="outer")

    # Agreement flag on the final label
    merged["agree"] = merged["edd_outcome_ollama"] == merged["edd_outcome_claude"]
    # Confidence delta (Claude − Ollama)
    merged["confidence_delta"] = (
        merged["confidence_score_claude"] - merged["confidence_score_ollama"]
    )

    # Reorder for readability
    merged = merged[[
        "row_number",
        "edd_outcome_ollama", "edd_outcome_claude", "agree",
        "confidence_score_ollama", "confidence_score_claude", "confidence_delta",
        "confidence_reason_ollama", "confidence_reason_claude",
    ]]

    # --- Summary stats ---
    both_valid = merged.dropna(subset=["edd_outcome_ollama", "edd_outcome_claude"])
    total      = len(merged)
    valid      = len(both_valid)
    agree_n    = int(both_valid["agree"].sum())
    agree_pct  = (agree_n / valid * 100) if valid else 0.0

    print(f"Total rows compared:        {total:,}")
    print(f"Rows with both valid labels: {valid:,}")
    print(f"Agreement:                   {agree_n:,} / {valid:,} ({agree_pct:.2f}%)")
    print(f"Disagreements:               {valid - agree_n:,}")

    if valid:
        # Confusion-style breakdown: Ollama (rows) vs Claude (cols)
        xtab = pd.crosstab(
            both_valid["edd_outcome_ollama"].astype(int),
            both_valid["edd_outcome_claude"].astype(int),
            rownames=["Ollama"], colnames=["Claude"],
            margins=True, margins_name="Total",
        )
        print("\nConfusion matrix (Ollama rows vs Claude cols):")
        print(xtab)

        mean_delta = both_valid["confidence_delta"].mean()
        print(f"\nMean confidence delta (Claude − Ollama): {mean_delta:+.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nSide-by-side comparison written to {out_path}")
    return merged