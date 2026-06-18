"""Phase 2 — normalize, dedupe, and unify all raw sources into reviews.parquet.

Schema: id | source | text | rating | date | score | url
Run:    python src/clean.py
Output: data/clean/reviews.parquet
"""

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "clean" / "reviews.parquet"

MIN_CHARS = 15

# ── text cleaning ─────────────────────────────────────────────────────────────

_ZW_RE       = re.compile(r"[​-‍‎‏﻿­]")
_HTML_ENT_RE = re.compile(r"&(amp|lt|gt|quot|apos|#\d+);", re.IGNORECASE)
_REDDIT_QT   = re.compile(r"(?m)^&gt;\s?|^>\s?")
_URL_ONLY    = re.compile(r"(?m)^\s*https?://\S+\s*$")
_DELETED     = re.compile(r"^\[(deleted|removed)\]$", re.MULTILINE)
_MULTI_SP    = re.compile(r"[ \t]+")
_MULTI_NL    = re.compile(r"\n{3,}")
_HAS_ALPHA   = re.compile(r"[a-zA-Z]")
_NON_ASCII   = re.compile(r"[^\x00-\x7F]")

_HTML_MAP = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _ent(m: re.Match) -> str:
    k = m.group(1).lower()
    if k.startswith("#"):
        try:
            return chr(int(k[1:]))
        except ValueError:
            return m.group(0)
    return _HTML_MAP.get(k, m.group(0))


def _clean_text(raw: str | None, source: str) -> str:
    if not raw:
        return ""
    t = str(raw)
    t = _ZW_RE.sub("", t)
    t = _HTML_ENT_RE.sub(_ent, t)
    t = t.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    t = unicodedata.normalize("NFC", t)

    if source == "reddit":
        t = _DELETED.sub("", t)
        t = _REDDIT_QT.sub("", t)
        t = _URL_ONLY.sub("", t)

    # Collapse multiple spaces/tabs (keep newlines for paragraph structure)
    t = _MULTI_SP.sub(" ", t)
    # Collapse 3+ blank lines to 2
    t = _MULTI_NL.sub("\n\n", t)
    return t.strip()


def _stable_id(source: str, key: str) -> str:
    return hashlib.sha256(f"{source}:{key}".encode()).hexdigest()[:16]


def _parse_date(val) -> str | None:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(float(val), tz=timezone.utc).isoformat()
        s = str(val).strip()
        if not s:
            return None
        # "2026-06-17 17:21:58" (Play Store — assume UTC)
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
        # ISO 8601 variants
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except Exception:
        return None


def _dedup_key(text: str) -> str:
    """Lowercase, collapse all whitespace — used as dedup fingerprint."""
    return re.sub(r"\s+", " ", text.lower().strip())


# ── per-source loaders ────────────────────────────────────────────────────────

def load_playstore() -> list[dict]:
    raw = json.loads((RAW / "playstore.json").read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        text = _clean_text(r.get("content"), "playstore")
        rows.append({
            "id":     _stable_id("playstore", r.get("reviewId") or text),
            "source": "playstore",
            "text":   text,
            "rating": float(r["score"]) if r.get("score") is not None else None,
            "date":   _parse_date(r.get("at")),
            "score":  None,
            "url":    None,
        })
    return rows


def load_appstore() -> list[dict]:
    raw = json.loads((RAW / "appstore.json").read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        title = (r.get("title") or "").strip()
        body  = (r.get("text")  or "").strip()
        combined = f"{title}\n\n{body}" if (title and body) else (title or body)
        text = _clean_text(combined, "appstore")
        rows.append({
            "id":     _stable_id("appstore", r.get("id") or text),
            "source": "appstore",
            "text":   text,
            "rating": float(r["rating"]) if r.get("rating") else None,
            "date":   _parse_date(r.get("date")),
            "score":  None,
            "url":    None,
        })
    return rows


def load_reddit() -> list[dict]:
    raw = json.loads((RAW / "reddit.json").read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        title    = (r.get("title")    or "").strip()
        selftext = (r.get("selftext") or "").strip()
        combined = f"{title}\n\n{selftext}" if selftext else title
        text = _clean_text(combined, "reddit")
        rows.append({
            "id":     _stable_id("reddit", r.get("id") or text),
            "source": "reddit",
            "text":   text,
            "rating": None,
            "date":   _parse_date(r.get("created_utc")),
            "score":  float(r["score"]) if r.get("score") is not None else None,
            "url":    r.get("url"),
        })
    return rows


def load_forum() -> list[dict]:
    raw = json.loads((RAW / "forum.json").read_text(encoding="utf-8"))
    rows = []
    for r in raw:
        title = (r.get("title") or "").strip()
        body  = (r.get("body")  or "").strip()
        combined = f"{title}\n\n{body}" if body else title
        text = _clean_text(combined, "forum")
        url  = r.get("url")
        rows.append({
            "id":     _stable_id("forum", url or text),
            "source": "forum",
            "text":   text,
            "rating": None,
            "date":   None,
            "score":  float(r["kudos"]) if r.get("kudos") is not None else None,
            "url":    url,
        })
    return rows


# ── filtering ─────────────────────────────────────────────────────────────────

def _drop_reason(text: str) -> str | None:
    """Return a reason string if the row should be dropped, else None."""
    if not text or not text.strip():
        return "empty"
    t = text.strip()
    if len(t) < MIN_CHARS:
        return "too_short"
    if not _HAS_ALPHA.search(t):
        return "no_alpha"
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    loaders = [
        ("playstore", load_playstore),
        ("appstore",  load_appstore),
        ("reddit",    load_reddit),
        ("forum",     load_forum),
    ]

    all_rows: list[dict] = []
    stats: dict[str, dict] = {}

    for source, loader in loaders:
        raw_rows = loader()
        n_in = len(raw_rows)

        dropped: dict[str, int] = {}
        kept: list[dict] = []
        seen_text: set[str] = set()

        for row in raw_rows:
            reason = _drop_reason(row["text"])
            if reason:
                dropped[reason] = dropped.get(reason, 0) + 1
                continue

            key = _dedup_key(row["text"])
            if key in seen_text:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                continue
            seen_text.add(key)
            kept.append(row)

        stats[source] = {
            "in":      n_in,
            "out":     len(kept),
            "dropped": n_in - len(kept),
            "reasons": dropped,
        }
        all_rows.extend(kept)

    df = pd.DataFrame(all_rows, columns=["id", "source", "text", "rating", "date", "score", "url"])

    # Estimate non-English rows (heuristic: >30% non-ASCII chars)
    def _non_en(text: str) -> bool:
        if not isinstance(text, str) or not text:
            return False
        return len(_NON_ASCII.findall(text)) / max(len(text), 1) > 0.30

    n_non_en = df["text"].apply(_non_en).sum()

    df.to_parquet(OUT, index=False)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Source':<12} {'In':>6} {'Out':>6} {'Dropped':>8}  Reasons")
    print("-" * 65)
    total_in = total_out = 0
    for src, s in stats.items():
        reasons_str = ", ".join(f"{k}={v}" for k, v in s["reasons"].items())
        print(f"{src:<12} {s['in']:>6} {s['out']:>6} {s['dropped']:>8}  {reasons_str}")
        total_in  += s["in"]
        total_out += s["out"]
    print("-" * 65)
    print(f"{'TOTAL':<12} {total_in:>6} {total_out:>6} {total_in - total_out:>8}")
    print("=" * 65)
    print(f"Estimated non-English rows: {n_non_en} ({n_non_en/max(total_out,1)*100:.1f}%)")
    print(f"Output: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)\n")

    # ── sample rows ───────────────────────────────────────────────────────────
    for src in ["playstore", "appstore", "reddit", "forum"]:
        sub = df[df["source"] == src].head(5)
        print(f"\n{'-'*65}")
        print(f"SAMPLE -- {src.upper()} ({len(df[df['source']==src])} rows)")
        print(f"{'-'*65}")
        for _, row in sub.iterrows():
            snippet = row["text"][:120].replace("\n", " ")
            snippet_safe = snippet.encode("ascii", "replace").decode("ascii")
            print(f"  id={row['id']}  rating={row['rating']}  score={row['score']}")
            print(f"  date={row['date']}")
            print(f"  text: {snippet_safe!r}")
            print()


if __name__ == "__main__":
    main()
