"""
EDD (Enhanced Due Diligence) Narrative Classification Pipeline — OpenAI/ChatGPT
================================================================================
Uses the OpenAI API with JSON mode.
- PII-safe: only [row_number, narrative] reach the model
- Checkpointing every 500 rows, exponential backoff on rate limits
- Output: 4-column CSV (row_number, edd_outcome, confidence_score, confidence_reason)

Run:
    python edd_classification_pipeline_openai.py
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
BASE_DIR = "/Users/bodhit/GenAi - Narrative Classification"

INPUT_CSV_PATH               = f"{BASE_DIR}/edd_final.csv"
SAMPLED_CSV_PATH             = f"{BASE_DIR}/edd_sampled_with_row_number.csv"
OUTPUT_CSV_PATH_OPENAI       = f"{BASE_DIR}/edd_classifications_openai.csv"
CHECKPOINT_CSV_PATH_OPENAI   = f"{BASE_DIR}/edd_classifications_openai.checkpoint.csv"

# --- Column names in source CSV ---
DECISION_COLUMN  = "Decision"
NARRATIVE_COLUMN = "narrative"

# --- Sampling ---
SAMPLE_SIZE  = 1000
RANDOM_SEED  = 42

# --- OpenAI API ---
# Current model options (April 2026):
#   "gpt-5.4"        — flagship. $2.50 / $15 per 1M input/output tokens. Highest quality.
#   "gpt-5.4-mini"   — recommended default. ~$0.75 / $3. Great for structured classification.
#   "gpt-5.4-nano"   — cheapest. $0.20 / $1.25. May lose nuance on borderline cases.
#   "gpt-4.1"        — previous generation flagship. Still supported.
#   "o3" / "o4-mini" — reasoning models. Overkill (and expensive) for binary classification.
OPENAI_MODEL         = "gpt-5.4"

# Path to a plain-text file containing ONLY your OpenAI API key (no quotes, no "export",
# nothing else). Falls back to OPENAI_API_KEY environment variable if missing or empty.
OPENAI_API_KEY_FILE = f"{BASE_DIR}/openai_api.txt"


def _load_openai_api_key(file_path: str) -> str:
    """Prefer key from file; fall back to OPENAI_API_KEY env var."""
    try:
        p = Path(file_path)
        if p.exists():
            key = p.read_text().strip()
            if key:
                return key
    except Exception as e:
        print(f"[warn] Could not read {file_path}: {e}")
    return os.environ.get("OPENAI_API_KEY", "")


OPENAI_API_KEY       = _load_openai_api_key(OPENAI_API_KEY_FILE)
CHECKPOINT_INTERVAL  = 500
MAX_RETRIES          = 5
INITIAL_BACKOFF_SEC  = 2
OPENAI_MAX_TOKENS    = 500  # max tokens for the JSON response


# ==============================================================
# PROMPTS
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
# STEP 1 — Load CSV + random sample (stratified available but off)
# ==============================================================

def stratified_sample(df: pd.DataFrame, col: str, n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation — kept available even though not currently used."""
    counts = df[col].value_counts()
    total  = counts.sum()
    pieces = []
    for label, cnt in counts.items():
        prop = cnt / total
        k    = max(1, round(prop * n))
        k    = min(k, cnt)
        grp  = df[df[col] == label]
        pieces.append(grp.sample(n=k, random_state=seed))
    sampled = pd.concat(pieces, ignore_index=True)
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_and_sample(path: str, decision_col: str, n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    if NARRATIVE_COLUMN not in df.columns:
        raise KeyError(f"Column '{NARRATIVE_COLUMN}' not found. Available: {list(df.columns)}")

    if decision_col in df.columns:
        orig = df[decision_col].value_counts(normalize=True)
        print("\nOriginal Decision distribution (informational only):")
        for k, v in orig.items():
            print(f"  {k!s:<40} {v*100:6.2f}%")

    # Simple random sampling on rows with a valid narrative
    valid = df[df[NARRATIVE_COLUMN].notna() & df[NARRATIVE_COLUMN].astype(str).str.strip().ne("")]
    dropped = len(df) - len(valid)
    if dropped:
        print(f"\nExcluded {dropped:,} rows with empty/null narrative. {len(valid):,} eligible.")

    k = min(n, len(valid))
    sampled = valid.sample(n=k, random_state=seed).reset_index(drop=True)

    # To restore stratified sampling, replace the two lines above with:
    # sampled = stratified_sample(df, decision_col, n, seed)

    print(f"\nSampled {len(sampled):,} rows (target {n:,})")
    return sampled


# ==============================================================
# STEP 2 — Assign row_number + extract only the model input
# ==============================================================

def prepare_model_input(sampled_df: pd.DataFrame, narrative_col: str):
    if narrative_col not in sampled_df.columns:
        raise KeyError(f"Column '{narrative_col}' not found. Available: {list(sampled_df.columns)}")

    sampled = sampled_df.copy()
    sampled["row_number"] = range(1, len(sampled) + 1)
    narrative_df = sampled[["row_number", narrative_col]].rename(
        columns={narrative_col: "narrative"}
    ).reset_index(drop=True)
    return narrative_df, sampled


# ==============================================================
# STEP 3 — Shared helpers
# ==============================================================

def parse_model_json(text: str) -> dict:
    """Strip ```json fences if present, then parse."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    return json.loads(t)


def normalize_result(row_num: int, parsed: dict) -> dict:
    return {
        "row_number":        row_num,
        "edd_outcome":       int(parsed["edd_outcome"]),
        "confidence_score":  float(parsed["confidence_score"]),
        "confidence_reason": str(parsed["confidence_reason"]),
    }


def error_result(row_num: int, err) -> dict:
    msg = f"ERROR: {type(err).__name__}: {err}" if isinstance(err, Exception) else f"ERROR: {err}"
    return {
        "row_number":        row_num,
        "edd_outcome":       None,
        "confidence_score":  None,
        "confidence_reason": msg,
    }


# ==============================================================
# STEP 4 — OpenAI classification with checkpointing + backoff
# ==============================================================

def classify_with_openai(narrative_df: pd.DataFrame) -> pd.DataFrame:
    from openai import OpenAI
    try:
        from openai import RateLimitError, APIError, APIConnectionError
    except ImportError:
        # Extremely old SDK fallback
        from openai import RateLimitError, APIError
        APIConnectionError = APIError

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set (neither file nor environment variable)."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    # ---- Resume from checkpoint if present ----
    completed_rows: set = set()
    results: list = []
    if Path(CHECKPOINT_CSV_PATH_OPENAI).exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV_PATH_OPENAI)
        completed_rows = set(int(x) for x in ckpt["row_number"].tolist())
        results        = ckpt.to_dict("records")
        print(f"Resuming from checkpoint: {len(completed_rows):,} rows already done")

    pending = narrative_df[~narrative_df["row_number"].isin(completed_rows)].reset_index(drop=True)
    print(f"Classifying {len(pending):,} remaining rows with {OPENAI_MODEL}")

    for i, (_, row) in enumerate(tqdm(pending.iterrows(), total=len(pending), desc="OpenAI")):
        row_num   = int(row["row_number"])
        narrative = row["narrative"]

        parsed_or_err = None

        for attempt in range(MAX_RETRIES):
            try:
                if pd.isna(narrative) or not str(narrative).strip():
                    raise ValueError("Empty narrative")

                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(narrative=narrative)},
                    ],
                    # JSON mode forces the model to return valid JSON.
                    # Note: SYSTEM_PROMPT must mention "JSON" (it does).
                    response_format={"type": "json_object"},
                    max_completion_tokens=OPENAI_MAX_TOKENS,
                    temperature=0,  # deterministic output for reproducibility
                )
                text   = resp.choices[0].message.content
                parsed = parse_model_json(text)
                parsed_or_err = ("ok", parsed)
                break

            except RateLimitError:
                wait = INITIAL_BACKOFF_SEC * (2 ** attempt)
                tqdm.write(f"[Row {row_num}] Rate limited (attempt {attempt+1}/{MAX_RETRIES}). Sleeping {wait}s")
                time.sleep(wait)

            except (APIError, APIConnectionError) as e:
                wait = INITIAL_BACKOFF_SEC * (2 ** attempt)
                tqdm.write(f"[Row {row_num}] API error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Sleeping {wait}s")
                time.sleep(wait)

            except Exception as e:
                # Non-retryable (bad JSON, empty narrative, model refusal, etc.)
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
            pd.DataFrame(results).to_csv(CHECKPOINT_CSV_PATH_OPENAI, index=False)
            tqdm.write(f"[Checkpoint] Wrote {len(results):,} rows to {CHECKPOINT_CSV_PATH_OPENAI}")

    return pd.DataFrame(
        results,
        columns=["row_number", "edd_outcome", "confidence_score", "confidence_reason"],
    )


# ==============================================================
# MAIN
# ==============================================================

def main():
    Path(SAMPLED_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)

    sampled = load_and_sample(INPUT_CSV_PATH, DECISION_COLUMN, SAMPLE_SIZE, RANDOM_SEED)
    narrative_df, sampled_with_rn = prepare_model_input(sampled, NARRATIVE_COLUMN)

    sampled_with_rn.to_csv(SAMPLED_CSV_PATH, index=False)
    print(f"\nSaved sampled data + row_number -> {SAMPLED_CSV_PATH}")
    print(f"Model will see ONLY these columns: {list(narrative_df.columns)}")

    results_df = classify_with_openai(narrative_df)

    Path(OUTPUT_CSV_PATH_OPENAI).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV_PATH_OPENAI, index=False)
    n_err = results_df["edd_outcome"].isna().sum()
    print(f"\nDone. Classified {len(results_df):,} rows ({n_err:,} errors) -> {OUTPUT_CSV_PATH_OPENAI}")

    if Path(CHECKPOINT_CSV_PATH_OPENAI).exists():
        Path(CHECKPOINT_CSV_PATH_OPENAI).unlink()
        print(f"Removed checkpoint file {CHECKPOINT_CSV_PATH_OPENAI}")


if __name__ == "__main__":
    main()