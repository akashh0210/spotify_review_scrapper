"""One-shot fix: re-tag the ~250 Gemini-tagged rows on Groq.

Gemini rows (tagged_by in gemini-2.5-flash-lite / gemini-2.5-flash) showed
99.5% non_discovery — a systematic labelling failure, not plausible real data.
This script replaces them with Groq tags to give a clean single-provider corpus.

Selects rows by tagged_by, not tag_error. All other rows are preserved as-is.
Checkpoint: data/tagged/_retag_gemini_checkpoint.parquet (every 50 rows).

Run:  python src/retag_gemini.py
"""

import json
import os
import re
import time
from pathlib import Path

import groq as groq_sdk
import pandas as pd
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv(".env.local") or load_dotenv()

BASE       = Path(__file__).parent.parent
TAGGED     = BASE / "data" / "tagged" / "reviews_tagged.parquet"
CKPT       = BASE / "data" / "tagged" / "_retag_gemini_checkpoint.parquet"

MODEL_GROQ   = os.getenv("GROQ_TAGGING_MODEL", "llama-3.1-8b-instant")
PROVIDER_TAG = "groq-llama-3.1-8b"

GEMINI_PROVIDERS = {"gemini-2.5-flash-lite", "gemini-2.5-flash"}

BATCH_SIZE       = 10
CKPT_EVERY       = 50
MAX_TEXT_CHARS   = 350
INTER_CALL_SLEEP = 1.0

# ── closed lists ──────────────────────────────────────────────────────────────
VALID_THEMES = frozenset({
    "recommendation_repetition", "discovery_friction", "generic_recommendations",
    "discover_weekly_dailymix",  "autoplay_radio_loop", "no_control_or_intent",
    "filter_bubble", "wants_new_but_safe", "positive_discovery", "non_discovery",
})
VALID_SENTIMENTS = frozenset({"positive", "neutral", "negative"})
VALID_SEGMENTS   = frozenset({
    "casual", "power_user", "genre_explorer",
    "mood_context_listener", "podcast_listener", "unknown",
})

SYSTEM = """\
You are a Spotify music-app review classifier focused on music discovery and recommendation UX.

For each review return a JSON object with EXACTLY these fields:
{"themes":[],"sentiment":"","segment":"","discovery_related":true,"one_line":"","language":""}

ALLOWED THEMES (closed list - do not invent others):
recommendation_repetition   algorithm repeats songs user already knows
discovery_friction          hard to find new/unfamiliar music
generic_recommendations     suggestions feel generic or non-personalised
discover_weekly_dailymix    mentions Discover Weekly / Daily Mix / Release Radar
autoplay_radio_loop         autoplay or radio loops same songs
no_control_or_intent        cannot steer or signal taste to the algorithm
filter_bubble               stuck in echo chamber of same artists/genres
wants_new_but_safe          wants new music but only within comfort zone
positive_discovery          happy with discovery or recommendation features
non_discovery               billing, UI, bugs, ads, crashes, or unrelated

RULES:
* themes: 1-3 values from closed list ONLY; strip any not on the list
* discovery_related: true if any theme other than non_discovery is present
* sentiment: "positive" | "neutral" | "negative"
* segment: "casual" | "power_user" | "genre_explorer" | "mood_context_listener" | "podcast_listener" | "unknown"
* one_line: <=20 words, summarise what the USER says (not generic filler)
* language: "en" if primarily English, else "other"

You receive a numbered list of reviews. Return a JSON ARRAY with exactly one object per review, same order."""


# ── helpers (identical to tag.py / retag_errors.py) ──────────────────────────

def _build_prompt(batch: list[dict]) -> str:
    parts = [f"Tag these {len(batch)} reviews. Return a JSON array of exactly {len(batch)} objects.\n"]
    for i, row in enumerate(batch, 1):
        meta = []
        if pd.notna(row.get("rating")):
            meta.append(f"rating={int(row['rating'])}/5")
        if pd.notna(row.get("score")):
            meta.append(f"upvotes={int(row['score'])}")
        meta_s = f" [{', '.join(meta)}]" if meta else ""
        text   = (row.get("text") or "")[:MAX_TEXT_CHARS]
        parts.append(f"[{i}] source={row['source']}{meta_s}\n{text}")
    return "\n\n".join(parts)


def _extract_array(text: str) -> list:
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    start = text.find("[")
    if start == -1:
        obj_s = text.find("{")
        if obj_s != -1:
            depth, end = 0, -1
            for i, ch in enumerate(text[obj_s:], obj_s):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                return [json.loads(text[obj_s : end + 1])]
        raise ValueError("No JSON in response")
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[": depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unmatched '['")
    parsed = json.loads(text[start : end + 1])
    return parsed if isinstance(parsed, list) else [parsed]


def _validate(raw) -> dict:
    if isinstance(raw, str):
        raw = {"themes": [raw]}
    elif isinstance(raw, list):
        raw = {"themes": raw}
    themes = [t for t in (raw.get("themes") or []) if t in VALID_THEMES]
    if not themes:
        themes = ["non_discovery"]
    sent = raw.get("sentiment", "neutral")
    if sent not in VALID_SENTIMENTS:
        sent = "neutral"
    seg = raw.get("segment", "unknown")
    if seg not in VALID_SEGMENTS:
        seg = "unknown"
    lang = raw.get("language", "en")
    if lang not in ("en", "other"):
        lang = "en"
    return {
        "themes":            themes,
        "sentiment":         sent,
        "segment":           seg,
        "discovery_related": any(t != "non_discovery" for t in themes),
        "one_line":          str(raw.get("one_line") or "")[:200],
        "language":          lang,
        "tag_error":         False,
        "tagged_by":         PROVIDER_TAG,
    }


@retry(
    retry=retry_if_exception_type(groq_sdk.RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=90),
    stop=stop_after_attempt(8),
    reraise=True,
)
def _api_call(client: groq_sdk.Groq, batch: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL_GROQ,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": _build_prompt(batch)},
        ],
        timeout=45,
    )
    return resp.choices[0].message.content


def _tag_batch(client: groq_sdk.Groq, batch: list[dict]) -> list[dict]:
    for attempt in range(2):
        try:
            content  = _api_call(client, batch)
            raw_list = _extract_array(content)
            if len(raw_list) != len(batch):
                raise ValueError(f"expected {len(batch)}, got {len(raw_list)}")
            return [_validate(r) for r in raw_list]
        except groq_sdk.RateLimitError:
            raise
        except Exception as exc:
            if attempt == 0:
                print(f"\n  [retry-parse] {exc}", flush=True)
                time.sleep(2)
            else:
                print(f"\n  [batch-fail] row-by-row: {exc}", flush=True)

    results: list[dict] = []
    for row in batch:
        try:
            raw_list = _extract_array(_api_call(client, [row]))
            results.append(_validate(raw_list[0]))
        except Exception as exc:
            print(f"\n  [row-fallback] id={row['id']} -> tag_error ({exc})", flush=True)
            results.append({
                "themes": ["non_discovery"], "sentiment": "neutral", "segment": "unknown",
                "discovery_related": False, "one_line": "", "language": "en",
                "tag_error": True, "tagged_by": PROVIDER_TAG,
            })
        time.sleep(0.5)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GROQ_API_KEY not set in .env.local")

    client = groq_sdk.Groq(api_key=api_key)

    df         = pd.read_parquet(TAGGED)
    gemini_mask = df["tagged_by"].isin(GEMINI_PROVIDERS)
    keep_df    = df[~gemini_mask].copy()   # all non-Gemini rows — untouched
    retag_df   = df[gemini_mask].copy()    # Gemini rows to replace
    n_target   = len(retag_df)

    print(f"Loaded {len(df)} rows from {TAGGED.name}")
    print(f"Rows to keep (non-Gemini) : {len(keep_df)}")
    print(f"Rows to re-tag (Gemini)   : {n_target}")
    print()
    print("Gemini rows by source:")
    for src, grp in retag_df.groupby("source"):
        disc = grp["discovery_related"].mean() * 100
        print(f"  {src:<12} {len(grp):>4}  disc_related={disc:.1f}%")
    print()

    # Resume from checkpoint
    retagged: list[dict] = []
    done_ids: set[str]   = set()
    if CKPT.exists():
        ckpt_df  = pd.read_parquet(CKPT)
        # Only resume rows still in the Gemini set (idempotent)
        valid    = ckpt_df[ckpt_df["id"].isin(set(retag_df["id"]))]
        retagged = valid.to_dict("records")
        done_ids = {r["id"] for r in retagged}
        if done_ids:
            print(f"Checkpoint: {len(done_ids)} rows already re-tagged. Resuming...")

    remaining   = retag_df[~retag_df["id"].isin(done_ids)].reset_index(drop=True)
    n_remaining = len(remaining)
    n_batches   = (n_remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Model   : {MODEL_GROQ}")
    print(f"To tag  : {n_remaining} in {n_batches} batches of {BATCH_SIZE}")
    print(f"Est. tokens: ~{n_remaining * 300:,} input + ~{n_remaining * 100:,} output\n")

    t0        = time.time()
    rows_done = 0

    for b_idx in range(n_batches):
        start_i = b_idx * BATCH_SIZE
        end_i   = min(start_i + BATCH_SIZE, n_remaining)
        batch   = remaining.iloc[start_i:end_i].to_dict("records")

        tags = _tag_batch(client, batch)
        for row, tag in zip(batch, tags):
            retagged.append({**row, **tag})
            rows_done += 1

        n_done   = len(done_ids) + rows_done
        elapsed  = time.time() - t0
        rate     = rows_done / max(elapsed, 1)
        eta_s    = (n_remaining - rows_done) / max(rate, 0.001)
        n_err    = sum(1 for r in retagged if r.get("tag_error"))
        print(
            f"  [{n_done}/{n_target}] batch {b_idx+1}/{n_batches} | "
            f"{elapsed/60:.1f}m elapsed | ETA {eta_s/60:.1f}m | errors={n_err}",
            flush=True,
        )

        if rows_done % CKPT_EVERY == 0:
            pd.DataFrame(retagged).to_parquet(CKPT, index=False)

        time.sleep(INTER_CALL_SLEEP)

    pd.DataFrame(retagged).to_parquet(CKPT, index=False)

    # Merge: preserved rows + newly Groq-tagged rows
    combined = pd.concat([keep_df, pd.DataFrame(retagged)], ignore_index=True)
    combined.to_parquet(TAGGED, index=False)

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total/60:.1f} min. Written {len(combined)} rows -> {TAGGED}\n")

    # ── summary ───────────────────────────────────────────────────────────────
    out = combined
    n   = len(out)
    n_err    = int(out["tag_error"].sum())
    n_disc   = int(out["discovery_related"].sum())

    print("=" * 62)
    print(f"Total rows         : {n}")
    print(f"discovery_related  : {n_disc} ({n_disc/n*100:.1f}%) True  |  {n-n_disc} ({(n-n_disc)/n*100:.1f}%) False")
    print(f"tag_error          : {n_err} ({n_err/n*100:.1f}%)")

    print("\ntagged_by (should be Groq-only + any tag_error rows):")
    for prov, grp in out.groupby("tagged_by"):
        errs = int(grp["tag_error"].sum())
        disc = grp["discovery_related"].mean() * 100
        print(f"  {prov:<28} {len(grp):>5} rows | disc={disc:.1f}% | errors={errs}")

    all_themes = [t for lst in out["themes"] for t in lst]
    theme_ser  = pd.Series(all_themes).value_counts()
    print(f"\nTheme frequency -- ALL ({len(all_themes)} tags across {n} rows):")
    for theme, cnt in theme_ser.items():
        rows_with = out["themes"].apply(lambda t: theme in t).sum()
        print(f"  {theme:<30} {cnt:>5}  ({rows_with/n*100:.1f}% of rows)")


if __name__ == "__main__":
    main()
