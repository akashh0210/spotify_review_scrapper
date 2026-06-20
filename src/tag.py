"""Phase 3 — tag every row with Groq llama-3.1-8b-instant (temp=0).

Outputs data/tagged/reviews_tagged.parquet with all original columns plus:
  themes[] | sentiment | segment | discovery_related | one_line | language | tag_error

Strategy:
- Loads the existing checkpoint (playstore all + appstore partial 1370/2340).
- From the UNTAGGED remainder, tags ONLY reddit + forum rows.
  Untagged appstore rows are intentionally skipped (1370 is sufficient signal).
- 10 reviews per API call (batch prompt → JSON array response)
- Tenacity exponential backoff on RateLimitError (429)
- Checkpoint to _checkpoint.parquet every 50 rows — fully resumable
- On JSON parse failure: retry once, then write FALLBACK and continue
- Closed theme list enforced server-side AND client-side (invalid themes stripped)

Run:  python src/tag.py
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
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv(".env.local") or load_dotenv()  # .env.local takes priority over .env

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent.parent
CLEAN  = BASE / "data" / "clean"  / "reviews.parquet"
TAGGED = BASE / "data" / "tagged" / "reviews_tagged.parquet"
CKPT   = BASE / "data" / "tagged" / "_checkpoint.parquet"

# ── config ────────────────────────────────────────────────────────────────────
MODEL         = os.getenv("GROQ_TAGGING_MODEL", "llama-3.1-8b-instant")
BATCH_SIZE    = 10
CKPT_EVERY    = 50   # checkpoint every N completed rows
MAX_TEXT_CHARS = 350  # truncate text in prompt (saves tokens)
INTER_BATCH_SLEEP = 1.0  # seconds between batches (polite pacing)

# ── closed lists (client-side enforcement) ────────────────────────────────────
VALID_THEMES = frozenset({
    "recommendation_repetition",
    "discovery_friction",
    "generic_recommendations",
    "discover_weekly_dailymix",
    "autoplay_radio_loop",
    "no_control_or_intent",
    "filter_bubble",
    "wants_new_but_safe",
    "positive_discovery",
    "non_discovery",
})
VALID_SENTIMENTS = frozenset({"positive", "neutral", "negative"})
VALID_SEGMENTS   = frozenset({
    "casual", "power_user", "genre_explorer",
    "mood_context_listener", "podcast_listener", "unknown",
})

FALLBACK_TAG: dict = {
    "themes":            ["non_discovery"],
    "sentiment":         "neutral",
    "segment":           "unknown",
    "discovery_related": False,
    "one_line":          "",
    "language":          "en",
    "tag_error":         True,
}

# ── system prompt ─────────────────────────────────────────────────────────────
SYSTEM = """\
You are a Spotify music-app review classifier focused on music discovery and recommendation UX.

For each review return a JSON object with EXACTLY these fields:
{"themes":[],"sentiment":"","segment":"","discovery_related":true,"one_line":"","language":""}

ALLOWED THEMES (closed list — do not invent others):
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
• themes: 1–3 values from closed list ONLY; strip any not on the list
• discovery_related: true if any theme other than non_discovery is present
• sentiment: "positive" | "neutral" | "negative"
• segment: "casual" | "power_user" | "genre_explorer" | "mood_context_listener" | "podcast_listener" | "unknown"
• one_line: ≤20 words, summarise what the USER says (not generic filler)
• language: "en" if primarily English, else "other"

You receive a numbered list of reviews. Return a JSON ARRAY with exactly one object per review, same order."""


# ── helpers ───────────────────────────────────────────────────────────────────

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
        # Bare object — wrap it
        obj_start = text.find("{")
        if obj_start != -1:
            # Find matching } using depth counting
            depth, end = 0, -1
            for i, ch in enumerate(text[obj_start:], obj_start):
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                return [json.loads(text[obj_start : end + 1])]
        raise ValueError("No JSON array or object in response")
    # Find matching ] using depth counting (avoids rfind picking up trailing brackets)
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "[": depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unmatched '[' in response")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        parsed = [parsed]
    return parsed


def _validate(raw) -> dict:
    # Coerce unexpected model outputs gracefully
    if isinstance(raw, str):
        raw = {"themes": [raw]}
    elif isinstance(raw, list):
        # Model returned just the themes list
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

    disc = any(t != "non_discovery" for t in themes)

    return {
        "themes":            themes,
        "sentiment":         sent,
        "segment":           seg,
        "discovery_related": disc,
        "one_line":          str(raw.get("one_line") or "")[:200],
        "language":          lang,
        "tag_error":         False,
    }


# ── API call (retried on rate-limit only) ─────────────────────────────────────

@retry(
    retry=retry_if_exception_type(groq_sdk.RateLimitError),
    wait=wait_exponential(multiplier=2, min=5, max=90),
    stop=stop_after_attempt(8),
    reraise=True,
)
def _api_call(client: groq_sdk.Groq, batch: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": _build_prompt(batch)},
        ],
        timeout=45,
    )
    return resp.choices[0].message.content


def _tag_batch(client: groq_sdk.Groq, batch: list[dict]) -> list[dict]:
    """Tag a batch; on parse/count error retry once; then fall back row-by-row."""
    for attempt in range(2):
        try:
            content  = _api_call(client, batch)
            raw_list = _extract_array(content)
            if not isinstance(raw_list, list):
                raise ValueError("response is not a list")
            if len(raw_list) != len(batch):
                raise ValueError(f"expected {len(batch)} items, got {len(raw_list)}")
            return [_validate(r) for r in raw_list]
        except groq_sdk.RateLimitError:
            raise  # already retried by decorator
        except Exception as exc:
            if attempt == 0:
                print(f"\n  [retry-parse] {exc}", flush=True)
                time.sleep(2)
            else:
                print(f"\n  [batch-fail] falling back to row-by-row: {exc}", flush=True)

    # Row-by-row fallback
    results: list[dict] = []
    for row in batch:
        try:
            content  = _api_call(client, [row])
            raw_list = _extract_array(content)
            results.append(_validate(raw_list[0]))
        except Exception as exc:
            print(f"\n  [row-fallback] id={row['id']} -> tag_error ({exc})", flush=True)
            results.append(dict(FALLBACK_TAG))
        time.sleep(0.5)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GROQ_API_KEY not set. Add it to .env")

    client = groq_sdk.Groq(api_key=api_key)
    TAGGED.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CLEAN)

    # ── resume from checkpoint ────────────────────────────────────────────────
    tagged_rows: list[dict] = []
    done_ids: set[str] = set()
    if CKPT.exists():
        ckpt_df     = pd.read_parquet(CKPT)
        tagged_rows = ckpt_df.to_dict("records")
        done_ids    = {r["id"] for r in tagged_rows}
        print(f"Checkpoint found: {len(done_ids)} rows already tagged.")

    # Tag ONLY reddit + forum from the untagged remainder.
    # Untagged appstore rows are intentionally skipped — 1370 tagged is enough.
    remaining = df[
        (~df["id"].isin(done_ids)) &
        (df["source"].isin(["reddit", "forum"]))
    ].reset_index(drop=True)

    n_ckpt      = len(done_ids)
    n_remaining = len(remaining)
    n_final     = n_ckpt + n_remaining   # expected rows in final parquet
    n_batches   = (n_remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Model  : {MODEL}")
    print(f"Sources: reddit + forum only (appstore partial kept as-is from checkpoint)")
    print(f"To tag : {n_remaining}  |  checkpoint: {n_ckpt}  |  final total: ~{n_final}")
    print(f"Batches: {n_batches} x {BATCH_SIZE}")
    print(f"Checkpoint every {CKPT_EVERY} rows -> {CKPT.name}\n")

    t0 = time.time()
    rows_this_run = 0

    for b_idx in range(n_batches):
        start_i = b_idx * BATCH_SIZE
        end_i   = min(start_i + BATCH_SIZE, n_remaining)
        batch   = remaining.iloc[start_i:end_i].to_dict("records")

        tags = _tag_batch(client, batch)

        for row, tag in zip(batch, tags):
            tagged_rows.append({**row, **tag})
            rows_this_run += 1

        # Progress
        n_done   = n_ckpt + rows_this_run
        elapsed  = time.time() - t0
        rate     = rows_this_run / max(elapsed, 1)
        eta_s    = (n_remaining - rows_this_run) / max(rate, 0.001)
        n_errors = sum(1 for r in tagged_rows if r.get("tag_error"))
        print(
            f"  [{n_done}/{n_final}] batch {b_idx+1}/{n_batches} | "
            f"{elapsed/60:.1f}m elapsed | ETA {eta_s/60:.1f}m | errors={n_errors}",
            flush=True,
        )

        # Checkpoint
        if rows_this_run % CKPT_EVERY == 0:
            pd.DataFrame(tagged_rows).to_parquet(CKPT, index=False)

        time.sleep(INTER_BATCH_SLEEP)

    # Final checkpoint + output
    out_df = pd.DataFrame(tagged_rows)
    out_df.to_parquet(CKPT,   index=False)
    out_df.to_parquet(TAGGED, index=False)

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total/60:.1f} min. Output: {TAGGED}\n")

    # ── report ────────────────────────────────────────────────────────────────
    n_err    = int(out_df["tag_error"].sum())
    n_non_en = int((out_df["language"] == "other").sum())
    n_disc   = int(out_df["discovery_related"].sum())
    n        = len(out_df)

    print("=" * 62)
    print(f"Total rows    : {n}  (playstore all + appstore partial 1370/2340 + reddit+forum all)")
    print(f"tag_error     : {n_err}  ({n_err/n*100:.1f}%)")
    print(f"language=other: {n_non_en}  ({n_non_en/n*100:.1f}%)")
    print(f"discovery_related True : {n_disc} ({n_disc/n*100:.1f}%)")
    print(f"discovery_related False: {n-n_disc} ({(n-n_disc)/n*100:.1f}%)")

    print("\nPer-source breakdown:")
    for src, grp in out_df.groupby("source"):
        errs = int(grp["tag_error"].sum())
        print(f"  {src:<12} {len(grp):>5} rows | errors={errs} ({errs/len(grp)*100:.1f}%)")

    print("\nSentiment:")
    for val, cnt in out_df["sentiment"].value_counts().items():
        print(f"  {val:<10} {cnt:>5}  ({cnt/n*100:.1f}%)")

    print("\nSegment:")
    for val, cnt in out_df["segment"].value_counts().items():
        print(f"  {val:<25} {cnt:>5}  ({cnt/n*100:.1f}%)")

    all_themes = [t for lst in out_df["themes"] for t in lst]
    theme_ser  = pd.Series(all_themes).value_counts()
    print(f"\nTheme frequency ({len(all_themes)} tags across {n} rows):")
    for theme, cnt in theme_ser.items():
        print(f"  {theme:<30} {cnt:>5}  ({cnt/n*100:.1f}%)")

    # 5 sample tag_error rows
    err_sample = out_df[out_df["tag_error"]].head(5)
    print(f"\n5 sample tag_error rows:")
    for _, row in err_sample.iterrows():
        txt = (row["text"] or "")[:80].encode("ascii", "replace").decode()
        print(f"  [{row['source']}] {row['id']} | {txt!r}")

    # 8 sample tagged rows (2 per source, non-error preferred)
    print(f"\n{'='*62}")
    print("SAMPLES (2 per source):")
    for src in ["playstore", "appstore", "reddit", "forum"]:
        sub = out_df[(out_df["source"] == src) & (~out_df["tag_error"])].head(2)
        if sub.empty:
            sub = out_df[out_df["source"] == src].head(2)
        for _, row in sub.iterrows():
            txt = (row["text"] or "")[:90].encode("ascii", "replace").decode()
            ol  = str(row.get("one_line") or "")[:90].encode("ascii", "replace").decode()
            print(f"\n[{src}] id={row['id']}")
            print(f"  text    : {txt!r}")
            print(f"  themes  : {row['themes']}")
            print(f"  sentiment={row['sentiment']}  segment={row['segment']}  "
                  f"disc={row['discovery_related']}  lang={row['language']}  err={row['tag_error']}")
            print(f"  one_line: {ol!r}")


if __name__ == "__main__":
    main()
