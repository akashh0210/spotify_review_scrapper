"""
Re-runnable pipeline orchestrator. Points the full engine at a new source
and regenerates all insights end-to-end.

Supported sources:
  playstore  -- scrapes Google Play Store via google-play-scraper
  csv        -- imports a CSV file (must have a 'text' column)

Usage (CLI):
  python src/run_workflow.py --source playstore [--app-id com.spotify.music] [--count 1000]
  python src/run_workflow.py --source csv --path reviews.csv [--max-rows 2000]

Run-time note:
  The tagging step (Stage 3) uses Groq's free tier and is the bottleneck.
  Rough estimate: ~2 min per 100 rows at 40 rows/min effective throughput.
  1,000 rows ≈ 20 min. Stay under ~1,500 rows to avoid hitting the daily budget.

What each run does:
  1. Clears data/raw/, data/clean/, data/tagged/, chroma_db/ (full reset)
  2. Scrapes Play Store OR writes CSV data to data/raw/
  3. clean.py  -> data/clean/reviews.parquet
  4. tag.py    -> data/tagged/reviews_tagged.parquet  (TAG_ALL_SOURCES=1)
  5. embed.py  -> chroma_db/  (rebuilds from scratch)
  6. aggregate.py -> data/insights/summary.json
  7. rag.py    -> data/insights/answers.json
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))

load_dotenv(str(BASE / ".env.local"))
load_dotenv(str(BASE / ".env"))

# ── directories ────────────────────────────────────────────────────────────────
RAW_DIR      = BASE / "data" / "raw"
CLEAN_DIR    = BASE / "data" / "clean"
TAGGED_DIR   = BASE / "data" / "tagged"
INSIGHTS_DIR = BASE / "data" / "insights"
CHROMA_DIR   = BASE / "chroma_db"

# ── safety limits ─────────────────────────────────────────────────────────────
MAX_ROWS_WARN  = 1_500   # warn above this (Groq free-tier daily budget)
MAX_CSV_ROWS   = 2_000   # hard cap for CSV uploads

TOTAL_STAGES = 7


# ── helpers ───────────────────────────────────────────────────────────────────

def _step(name: str, stage: int, cb: Callable | None) -> None:
    if cb:
        cb(name, stage, TOTAL_STAGES)
    else:
        print(f"\n[{stage}/{TOTAL_STAGES}] {name} ...", flush=True)


def _clear_pipeline_artifacts() -> None:
    """Delete all regenerable outputs so each workflow run starts clean."""
    for d in [RAW_DIR, CLEAN_DIR, TAGGED_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    # Rebuild ChromaDB from scratch — old collection would have stale vectors
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def _scrape_playstore(app_id: str, count: int) -> int:
    """Scrape Play Store and write to data/raw/playstore.json. Returns item count."""
    os.environ["PLAYSTORE_APP_ID"]       = app_id
    os.environ["PLAYSTORE_REVIEW_COUNT"] = str(count)

    import scrape_playstore as sp
    importlib.reload(sp)   # pick up env var changes
    sp.main()
    return len(json.loads((RAW_DIR / "playstore.json").read_text(encoding="utf-8")))


def _import_csv(path: str, max_rows: int = MAX_CSV_ROWS) -> int:
    """Parse a CSV and write to data/raw/csv_input.json. Returns row count."""
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise ValueError(
            "CSV must contain a 'text' column. "
            f"Found columns: {list(df.columns)}"
        )

    if len(df) > max_rows:
        print(f"  [warn] CSV has {len(df)} rows — truncating to {max_rows} "
              f"to stay within Groq free-tier budget.")
        df = df.head(max_rows)

    records = []
    for i, row in df.iterrows():
        records.append({
            "id":     str(row.get("id", f"csv_{i}")),
            "text":   str(row.get("text", "")),
            "rating": row.get("rating") if "rating" in df.columns else None,
            "date":   str(row.get("date", "")) if "date" in df.columns else None,
            "source": str(row.get("source", "csv")) if "source" in df.columns else "csv",
            "score":  row.get("score") if "score" in df.columns else None,
            "url":    str(row.get("url", "")) if "url" in df.columns else None,
        })

    out = RAW_DIR / "csv_input.json"
    out.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    return len(records)


# ── main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    source: str,
    app_id: str = "com.spotify.music",
    count: int = 1_000,
    csv_path: str | None = None,
    max_csv_rows: int = MAX_CSV_ROWS,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict:
    """
    Run the full pipeline for the given source. Returns a summary dict.

    Args:
        source:            "playstore" | "csv"
        app_id:            Play Store app ID (playstore only)
        count:             Reviews to scrape (playstore only)
        csv_path:          Path to CSV file (csv only)
        max_csv_rows:      Cap on CSV rows processed
        progress_callback: Optional fn(stage_name, current, total) for UI updates

    Returns:
        dict with raw_count, clean_count, disc_pct, top_themes, elapsed_s
    """
    t0 = time.time()

    def _p(name: str, stage: int) -> None:
        _step(name, stage, progress_callback)

    # Stage 0 — clear
    _p("Clearing previous pipeline artifacts", 0)
    _clear_pipeline_artifacts()

    # Stage 1 — scrape / import
    if source == "playstore":
        _p(f"Scraping Play Store: {app_id} ({count} reviews)", 1)
        raw_count = _scrape_playstore(app_id, count)
        print(f"  Scraped {raw_count} Play Store reviews", flush=True)
    elif source == "csv":
        if not csv_path:
            raise ValueError("csv_path is required for source='csv'")
        _p(f"Importing CSV: {Path(csv_path).name}", 1)
        raw_count = _import_csv(csv_path, max_csv_rows)
        print(f"  Imported {raw_count} rows from CSV", flush=True)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'playstore' or 'csv'.")

    if raw_count > MAX_ROWS_WARN:
        print(
            f"  [warn] {raw_count} rows will generate ~{raw_count * 150:,} tokens. "
            "May hit Groq free-tier daily limit (tagging could take 30-60 min).",
            flush=True,
        )

    # Stage 2 — clean
    _p("Cleaning & deduplicating", 2)
    import clean as clean_mod
    importlib.reload(clean_mod)
    clean_mod.main()

    # Stage 3 — tag (all sources for a workflow run)
    _p("Tagging with Groq llama-3.1-8b-instant (slow step — ~2 min per 100 rows)", 3)
    os.environ["TAG_ALL_SOURCES"] = "1"
    # Remove stale checkpoint so we tag fresh
    for ckpt in TAGGED_DIR.glob("_checkpoint*.parquet"):
        ckpt.unlink(missing_ok=True)
    import tag as tag_mod
    importlib.reload(tag_mod)
    tag_mod.main()
    del os.environ["TAG_ALL_SOURCES"]

    # Stage 4 — embed
    _p("Building ChromaDB vector index", 4)
    import embed as embed_mod
    importlib.reload(embed_mod)
    embed_mod.main()

    # Stage 5 — aggregate
    _p("Computing theme aggregates & unmet-needs weights", 5)
    import aggregate as aggregate_mod
    importlib.reload(aggregate_mod)
    aggregate_mod.main()

    # Stage 6 — RAG answers
    _p("Answering the six discovery questions (Groq 70B)", 6)
    import rag as rag_mod
    importlib.reload(rag_mod)
    rag_mod.main()

    elapsed = time.time() - t0

    # Build summary
    summary_path = INSIGHTS_DIR / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    ov = summary.get("overview", {})
    tf = summary.get("theme_frequency", {})
    top_themes = sorted(
        [(t, v["count"]) for t, v in tf.items() if t != "non_discovery"],
        key=lambda x: x[1], reverse=True,
    )[:5]

    return {
        "source":      source,
        "raw_count":   raw_count,
        "clean_count": ov.get("total_reviews", 0),
        "disc_pct":    ov.get("discovery_related", {}).get("pct", 0.0),
        "top_themes":  top_themes,
        "elapsed_s":   round(elapsed, 1),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Spotify Discovery Review Engine pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source", choices=["playstore", "csv"], required=True,
        help="Data source to analyze",
    )
    parser.add_argument("--app-id",   default="com.spotify.music",
                        help="Play Store app ID (--source playstore)")
    parser.add_argument("--count",    type=int, default=1_000,
                        help="Number of Play Store reviews to scrape")
    parser.add_argument("--path",     help="CSV file path (--source csv)")
    parser.add_argument("--max-rows", type=int, default=MAX_CSV_ROWS,
                        help=f"Max CSV rows (default {MAX_CSV_ROWS}, Groq budget cap)")
    args = parser.parse_args()

    print("=" * 58)
    print("Spotify Discovery Review Engine — Workflow Run")
    print("=" * 58)
    print(f"Source : {args.source}")
    if args.source == "playstore":
        print(f"App ID : {args.app_id}")
        print(f"Count  : {args.count} reviews")
    else:
        print(f"CSV    : {args.path}")
        print(f"Max    : {args.max_rows} rows")
    print()

    result = run_pipeline(
        source=args.source,
        app_id=args.app_id,
        count=args.count,
        csv_path=args.path,
        max_csv_rows=args.max_rows,
    )

    print("\n" + "=" * 58)
    print("PIPELINE COMPLETE")
    print("=" * 58)
    print(f"Source       : {result['source']}")
    print(f"Raw items    : {result['raw_count']:,}")
    print(f"Clean rows   : {result['clean_count']:,}")
    print(f"Discovery    : {result['disc_pct']:.1f}%")
    print(f"Elapsed      : {result['elapsed_s']:.0f}s ({result['elapsed_s']/60:.1f} min)")
    print("\nTop 5 themes:")
    for theme, cnt in result["top_themes"]:
        print(f"  {theme.replace('_', ' ').title():<30} {cnt:>5}")
    print(f"\nInsights -> {INSIGHTS_DIR}/")
    print(f"Vectors  -> {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
