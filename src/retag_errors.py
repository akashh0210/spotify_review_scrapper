"""Re-tag all tag_error=True rows in reviews_tagged.parquet via Gemini.

Uses google-genai SDK (supports AQ.* auth keys).
Overwrites only error rows in-place; all good rows are preserved untouched.
Checkpoints every 50 rows to data/tagged/_retag_checkpoint.parquet.

Rate: 6 s between API calls (~10 RPM, within Gemini free-tier limit).
Model: gemini-2.0-flash (set MODEL below; gemini-2.5-flash-lite has higher RPD headroom).

Run:  python src/retag_errors.py
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv(".env.local") or load_dotenv()

BASE       = Path(__file__).parent.parent
TAGGED     = BASE / "data" / "tagged" / "reviews_tagged.parquet"
RETAG_CKPT = BASE / "data" / "tagged" / "_retag_checkpoint.parquet"

MODEL            = "gemini-1.5-flash"   # separate daily quota from gemini-2.0-flash
PROVIDER_TAG     = "gemini-1.5-flash"
BATCH_SIZE       = 10
CKPT_EVERY       = 50
INTER_CALL_SLEEP = 6.0   # 6 s = 10 RPM, within Gemini free-tier limit

# ── closed lists (same as tag.py) ─────────────────────────────────────────────
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

FALLBACK_TAG: dict = {
    "themes": ["non_discovery"], "sentiment": "neutral", "segment": "unknown",
    "discovery_related": False, "one_line": "", "language": "en",
    "tag_error": True, "tagged_by": PROVIDER_TAG,
}

# ── prompt (same taxonomy as tag.py) ──────────────────────────────────────────
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

MAX_TEXT_CHARS = 350


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
        obj_start = text.find("{")
        if obj_start != -1:
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


# ── Gemini API call with rate-limit retry ─────────────────────────────────────

def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "quota", "rate limit", "resource exhausted", "too many"))


@retry(
    retry=retry_if_exception(_is_rate_limited),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _api_call(client, batch: list[dict]) -> str:
    resp = client.models.generate_content(
        model=MODEL,
        contents=_build_prompt(batch),
        config={
            "temperature": 0,
            "response_mime_type": "application/json",
            "system_instruction": SYSTEM,
        },
    )
    return resp.text


def _tag_batch(client, batch: list[dict]) -> list[dict]:
    """Tag a batch; retry parse once; then fall back row-by-row."""
    for attempt in range(2):
        try:
            content  = _api_call(client, batch)
            raw_list = _extract_array(content)
            if len(raw_list) != len(batch):
                raise ValueError(f"expected {len(batch)}, got {len(raw_list)}")
            return [_validate(r) for r in raw_list]
        except Exception as exc:
            if attempt == 0:
                print(f"\n  [retry-parse] {exc}", flush=True)
                time.sleep(2)
            else:
                print(f"\n  [batch-fail] falling back to row-by-row: {exc}", flush=True)

    results: list[dict] = []
    for row in batch:
        try:
            content  = _api_call(client, [row])
            raw_list = _extract_array(content)
            results.append(_validate(raw_list[0]))
        except Exception as exc:
            print(f"\n  [row-fallback] id={row['id']} -> tag_error ({exc})", flush=True)
            fb = dict(FALLBACK_TAG)
            results.append(fb)
        time.sleep(2)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY not set in .env.local")

    from google import genai
    client = genai.Client(api_key=api_key)

    if not TAGGED.exists():
        raise SystemExit(f"ERROR: {TAGGED} not found — run tag.py first")

    # Load the full tagged parquet and split
    df       = pd.read_parquet(TAGGED)
    good_df  = df[~df["tag_error"]].copy()
    error_df = df[df["tag_error"]].copy()
    n_total_errors = len(error_df)

    print(f"Loaded {len(df)} rows from {TAGGED.name}")
    print(f"Good rows  : {len(good_df)}")
    print(f"Error rows : {n_total_errors}")
    print()
    print("Error breakdown by source:")
    for src, grp in error_df.groupby("source"):
        print(f"  {src:<12} {len(grp):>5} errors")
    print()

    # Resume from retag checkpoint
    retagged: list[dict] = []
    done_ids: set[str]   = set()
    if RETAG_CKPT.exists():
        ckpt_df  = pd.read_parquet(RETAG_CKPT)
        retagged = ckpt_df.to_dict("records")
        done_ids = {r["id"] for r in retagged}
        print(f"Retag checkpoint: {len(done_ids)} rows already re-tagged. Resuming...")

    remaining   = error_df[~error_df["id"].isin(done_ids)].reset_index(drop=True)
    n_remaining = len(remaining)
    n_batches   = (n_remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Model  : {MODEL}")
    print(f"To tag : {n_remaining} rows in {n_batches} batches of {BATCH_SIZE}")
    print(f"Rate   : {INTER_CALL_SLEEP}s between calls (~{60/INTER_CALL_SLEEP:.0f} RPM)")
    print(f"ETA    : ~{n_batches * INTER_CALL_SLEEP / 60:.0f} min\n")

    t0        = time.time()
    rows_done = 0

    for b_idx in range(n_batches):
        start_i = b_idx * BATCH_SIZE
        end_i   = min(start_i + BATCH_SIZE, n_remaining)
        batch   = remaining.iloc[start_i:end_i].to_dict("records")

        tags = _tag_batch(client, batch)

        for row, tag in zip(batch, tags):
            merged = {**row, **tag}
            retagged.append(merged)
            rows_done += 1

        # Progress
        n_done   = len(done_ids) + rows_done
        elapsed  = time.time() - t0
        rate     = rows_done / max(elapsed, 1)
        eta_s    = (n_remaining - rows_done) / max(rate, 0.001)
        n_err    = sum(1 for r in retagged if r.get("tag_error"))
        print(
            f"  [{n_done}/{n_total_errors}] batch {b_idx+1}/{n_batches} | "
            f"{elapsed/60:.1f}m elapsed | ETA {eta_s/60:.1f}m | still-error={n_err}",
            flush=True,
        )

        if rows_done % CKPT_EVERY == 0:
            pd.DataFrame(retagged).to_parquet(RETAG_CKPT, index=False)

        time.sleep(INTER_CALL_SLEEP)

    # Save final retag checkpoint
    pd.DataFrame(retagged).to_parquet(RETAG_CKPT, index=False)

    # Merge good rows + all retagged rows and write back
    retagged_df = pd.DataFrame(retagged)
    combined    = pd.concat([good_df, retagged_df], ignore_index=True)
    combined.to_parquet(TAGGED, index=False)

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total/60:.1f} min.")
    print(f"Written {len(combined)} rows -> {TAGGED}\n")

    # ── summary ───────────────────────────────────────────────────────────────
    out = combined
    n   = len(out)
    n_err    = int(out["tag_error"].sum())
    n_non_en = int((out["language"] == "other").sum())
    n_disc   = int(out["discovery_related"].sum())

    print("=" * 65)
    print(f"Total rows    : {n}")
    print(f"tag_error     : {n_err} ({n_err/n*100:.1f}%)")
    print(f"language=other: {n_non_en} ({n_non_en/n*100:.1f}%)")
    print(f"discovery_related True : {n_disc} ({n_disc/n*100:.1f}%)")
    print(f"discovery_related False: {n-n_disc} ({(n-n_disc)/n*100:.1f}%)")

    print("\nPer-source breakdown (errors after re-tag):")
    for src, grp in out.groupby("source"):
        errs = int(grp["tag_error"].sum())
        print(f"  {src:<12} {len(grp):>5} rows | errors={errs} ({errs/len(grp)*100:.1f}%)")

    print("\nPer-provider (tagged_by):")
    for prov, grp in out.groupby("tagged_by"):
        errs = int(grp["tag_error"].sum())
        print(f"  {prov:<25} {len(grp):>5} rows | errors={errs} ({errs/len(grp)*100:.1f}%)")

    print("\nSentiment:")
    for val, cnt in out["sentiment"].value_counts().items():
        print(f"  {val:<10} {cnt:>5}  ({cnt/n*100:.1f}%)")

    print("\nSegment:")
    for val, cnt in out["segment"].value_counts().items():
        print(f"  {val:<25} {cnt:>5}  ({cnt/n*100:.1f}%)")

    all_themes = [t for lst in out["themes"] for t in lst]
    theme_ser  = pd.Series(all_themes).value_counts()
    print(f"\nTheme frequency -- ALL ({len(all_themes)} tags across {n} rows):")
    for theme, cnt in theme_ser.items():
        print(f"  {theme:<30} {cnt:>5}  ({cnt/n*100:.1f}%)")

    for prov in sorted(out["tagged_by"].unique()):
        sub = out[out["tagged_by"] == prov]
        sub_themes = [t for lst in sub["themes"] for t in lst]
        if not sub_themes:
            continue
        sub_ser = pd.Series(sub_themes).value_counts()
        print(f"\nTheme frequency -- {prov} ({len(sub)} rows):")
        for theme, cnt in sub_ser.items():
            print(f"  {theme:<30} {cnt:>5}  ({cnt/len(sub)*100:.1f}%)")

    print(f"\n{'='*65}")
    print("SAMPLES (2 per source, non-error preferred):")
    for src in ["playstore", "appstore", "reddit", "forum"]:
        sub = out[(out["source"] == src) & (~out["tag_error"])].head(2)
        if sub.empty:
            sub = out[out["source"] == src].head(2)
        for _, row in sub.iterrows():
            txt = (row["text"] or "")[:90].encode("ascii", "replace").decode()
            ol  = str(row.get("one_line") or "")[:90].encode("ascii", "replace").decode()
            print(f"\n[{src}] id={row['id']}  tagged_by={row['tagged_by']}")
            print(f"  text    : {txt!r}")
            print(f"  themes  : {row['themes']}")
            print(f"  sentiment={row['sentiment']}  disc={row['discovery_related']}  "
                  f"lang={row['language']}  err={row['tag_error']}")
            print(f"  one_line: {ol!r}")


if __name__ == "__main__":
    main()
