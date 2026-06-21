"""Phase 5A — aggregate tagged reviews → data/insights/summary.json.

Reads reviews_tagged.parquet (excludes tag_error rows = 5,708 rows).
Sections: overview, theme_frequency, segment_x_theme (Q5), unmet_needs (Q6),
top_quotes.

Run:  python src/aggregate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE    = Path(__file__).parent.parent
TAGGED  = BASE / "data" / "tagged" / "reviews_tagged.parquet"
OUT     = BASE / "data" / "insights" / "summary.json"

DISCOVERY_THEMES = [
    "discovery_friction",
    "recommendation_repetition",
    "generic_recommendations",
    "discover_weekly_dailymix",
    "autoplay_radio_loop",
    "no_control_or_intent",
    "filter_bubble",
    "wants_new_but_safe",
    "positive_discovery",
]
SEGMENTS = [
    "casual", "power_user", "genre_explorer",
    "mood_context_listener", "podcast_listener", "unknown",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _has_theme(themes_val, theme: str) -> bool:
    """Safe membership check that handles numpy arrays, lists, and None."""
    if themes_val is None:
        return False
    try:
        return theme in list(themes_val)
    except TypeError:
        return False


def _safe(val, default=None):
    """Convert NaN / numpy scalars to Python native types."""
    if val is None:
        return default
    if isinstance(val, float) and val != val:   # NaN
        return default
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _quote(row, max_chars: int = 350) -> dict:
    return {
        "text":   (row.get("text") or "")[:max_chars],
        "source": str(row.get("source") or ""),
        "rating": _safe(row.get("rating")),
        "score":  _safe(row.get("score")),
    }


def _top_quotes(theme_rows: pd.DataFrame, n: int = 3) -> list[dict]:
    """Pick n representative quotes, preferring high-score items + source diversity."""
    if theme_rows.empty:
        return []
    df = theme_rows.copy()
    df["_s"] = df["score"].apply(lambda x: float(x) if pd.notna(x) and float(x) > 0 else 0.0)
    df = df.sort_values("_s", ascending=False)

    selected, seen_sources = [], set()
    # First pass: try to get source diversity
    for _, row in df.iterrows():
        if len(selected) >= n:
            break
        src = row["source"]
        text = (row.get("text") or "")
        if len(text) < 30:
            continue
        if src in seen_sources and len(df) > n * 2:
            continue
        seen_sources.add(src)
        selected.append(_quote(row.to_dict()))
    # Fill if still short
    for _, row in df.iterrows():
        if len(selected) >= n:
            break
        q = _quote(row.to_dict())
        if q not in selected and len(q["text"]) >= 30:
            selected.append(q)
    return selected[:n]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df_all  = pd.read_parquet(TAGGED)
    df      = df_all[~df_all["tag_error"]].reset_index(drop=True)
    n       = len(df)
    disc_df = df[df["discovery_related"]]

    print(f"Loaded {len(df_all)} rows; using {n} after dropping tag_error rows.")

    # ── 1. overview ───────────────────────────────────────────────────────────
    sources = {}
    for src in sorted(df["source"].unique()):
        sub   = df[df["source"] == src]
        dates = sub["date"].dropna().sort_values()
        sources[src] = {
            "count":      int(len(sub)),
            "pct":        round(len(sub) / n * 100, 1),
            "date_range": {
                "earliest": str(dates.iloc[0])  if len(dates) else None,
                "latest":   str(dates.iloc[-1]) if len(dates) else None,
            },
        }

    sentiment_split = {}
    for sent in ["positive", "neutral", "negative"]:
        cnt = int((df["sentiment"] == sent).sum())
        sentiment_split[sent] = {"count": cnt, "pct": round(cnt / n * 100, 1)}

    overview = {
        "total_reviews":     n,
        "sources":           sources,
        "discovery_related": {
            "count": int(len(disc_df)),
            "pct":   round(len(disc_df) / n * 100, 1),
        },
        "sentiment": sentiment_split,
    }

    # ── 2. theme_frequency ───────────────────────────────────────────────────
    theme_frequency = {}
    for theme in DISCOVERY_THEMES + ["non_discovery"]:
        mask      = df["themes"].apply(lambda t: _has_theme(t, theme))
        theme_rows = df[mask]
        cnt        = len(theme_rows)

        disc_mask  = disc_df["themes"].apply(lambda t: _has_theme(t, theme))
        disc_cnt   = int(disc_mask.sum())

        theme_frequency[theme] = {
            "count":             cnt,
            "pct_of_all":        round(cnt / n * 100, 1),
            "pct_of_discovery":  round(disc_cnt / max(len(disc_df), 1) * 100, 1),
            "top_quotes":        _top_quotes(theme_rows),
        }

    # ── 3. segment_x_theme (Q5) ───────────────────────────────────────────────
    segment_x_theme = {}
    for seg in SEGMENTS:
        seg_rows = df[df["segment"] == seg]
        if seg_rows.empty:
            continue

        theme_data = {}
        for theme in DISCOVERY_THEMES:
            cnt = int(seg_rows["themes"].apply(lambda t: _has_theme(t, theme)).sum())
            if cnt > 0:
                theme_data[theme] = {
                    "count": cnt,
                    "pct":   round(cnt / len(seg_rows) * 100, 1),
                }

        top_3 = sorted(theme_data.items(), key=lambda x: x[1]["count"], reverse=True)[:3]

        segment_x_theme[seg] = {
            "total":          int(len(seg_rows)),
            "discovery_rate": round(float(seg_rows["discovery_related"].mean()) * 100, 1),
            "top_themes":     [{"theme": t, **v} for t, v in top_3],
            "theme_rates":    theme_data,
        }

    # Which segment has highest discovery_friction rate
    friction_rates = {
        seg: data["theme_rates"].get("discovery_friction", {}).get("pct", 0.0)
        for seg, data in segment_x_theme.items()
    }
    highest_seg = max(friction_rates, key=friction_rates.get) if friction_rates else None
    segment_x_theme["_analysis"] = {
        "highest_discovery_friction": {
            "segment": highest_seg,
            "pct":     friction_rates.get(highest_seg, 0),
        }
    }

    # ── 4. unmet_needs (Q6) — score-weighted ─────────────────────────────────
    scored = df[df["score"].notna() & (df["score"].astype(float) > 0)].copy()
    scored["_score"] = scored["score"].astype(float)

    unmet_map = {}
    for theme in DISCOVERY_THEMES:
        mask       = scored["themes"].apply(lambda t: _has_theme(t, theme))
        theme_scored = scored[mask]
        if theme_scored.empty:
            continue
        w_score    = float(theme_scored["_score"].sum())
        top_item   = theme_scored.nlargest(1, "_score").iloc[0]
        unmet_map[theme] = {
            "weighted_score": round(w_score, 1),
            "item_count":     int(len(theme_scored)),
            "top_quote": {
                "text":   (top_item.get("text") or "")[:350],
                "source": str(top_item.get("source") or ""),
                "score":  float(top_item["_score"]),
                "rating": _safe(top_item.get("rating")),
            },
        }

    unmet_needs = sorted(unmet_map.items(), key=lambda x: x[1]["weighted_score"], reverse=True)[:5]
    unmet_needs_list = [{"theme": t, **v} for t, v in unmet_needs]

    # ── 5. top_quotes (3 per discovery theme) ────────────────────────────────
    top_quotes = {}
    for theme in DISCOVERY_THEMES:
        mask       = df["themes"].apply(lambda t: _has_theme(t, theme))
        theme_rows = df[mask]
        quotes     = []
        for _, row in theme_rows.sort_values(
            "score",
            ascending=False,
            key=lambda s: s.fillna(0).astype(float),
        ).iterrows():
            text = (row.get("text") or "")
            if len(text) < 30:
                continue
            quotes.append({
                "text":      text[:350],
                "source":    str(row.get("source") or ""),
                "rating":    _safe(row.get("rating")),
                "score":     _safe(row.get("score")),
                "sentiment": str(row.get("sentiment") or ""),
                "segment":   str(row.get("segment") or ""),
                "theme":     theme,
            })
            if len(quotes) >= 3:
                break
        top_quotes[theme] = quotes

    # ── write ─────────────────────────────────────────────────────────────────
    summary = {
        "overview":       overview,
        "theme_frequency": theme_frequency,
        "segment_x_theme": segment_x_theme,
        "unmet_needs":    unmet_needs_list,
        "top_quotes":     top_quotes,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── terminal summary ──────────────────────────────────────────────────────
    print(f"\nWritten -> {OUT}  ({OUT.stat().st_size // 1024} KB)\n")

    print("Top 5 themes by count:")
    by_count = sorted(
        [(t, v["count"]) for t, v in theme_frequency.items() if t != "non_discovery"],
        key=lambda x: x[1], reverse=True,
    )[:5]
    for t, c in by_count:
        print(f"  {t:<30} {c:>5}")

    a = segment_x_theme.get("_analysis", {}).get("highest_discovery_friction", {})
    print(f"\nHighest discovery_friction segment: {a.get('segment')} ({a.get('pct')}%)")

    print("\nTop 3 unmet needs (score-weighted):")
    for u in unmet_needs_list[:3]:
        print(f"  {u['theme']:<30} weighted={u['weighted_score']:.0f}  items={u['item_count']}")


if __name__ == "__main__":
    main()
