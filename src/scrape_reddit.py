"""Scrape Spotify discovery threads from Reddit via Pullpush.io (no auth required).

Pullpush.io is the community-run Pushshift replacement that indexes all public
Reddit content. No auth, no OAuth. Fetches submission metadata (title + selftext)
only — no per-post comment API calls, which are slow and redundant at this volume.
2 s delay between searches + tenacity retry on 429/503.
"""

import json
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "reddit.json"

SUBREDDITS = ["spotify", "truespotify"]
SEARCH_TERMS = [
    "discover weekly",
    "recommendations",
    "same songs",
    "repetitive",
    "algorithm",
    "new music",
    "daily mix",
    "autoplay",
    "stuck",
    "radio",
]
RESULTS_PER_SEARCH = 100
DELAY = 2.0

HEADERS = {
    "User-Agent": "spotify-discovery-research/0.1 (educational data analysis)",
    "Accept": "application/json",
}

PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _search_submissions(subreddit: str, term: str) -> list[dict]:
    resp = requests.get(
        f"{PULLPUSH_BASE}/submission/",
        headers=HEADERS,
        params={
            "subreddit": subreddit,
            "q": term,
            "size": RESULTS_PER_SEARCH,
            "sort": "desc",
            "sort_type": "score",
        },
        timeout=30,
    )
    if resp.status_code in (429, 503):
        raise requests.HTTPError(f"{resp.status_code} rate-limited")
    resp.raise_for_status()
    return resp.json().get("data", [])


def scrape() -> list[dict]:
    seen_ids: set[str] = set()
    results: list[dict] = []

    for sub in SUBREDDITS:
        for term in SEARCH_TERMS:
            print(f"  r/{sub} -- '{term}'...", end=" ", flush=True)
            try:
                submissions = _search_submissions(sub, term)
            except Exception as exc:
                print(f"failed: {exc}")
                time.sleep(DELAY)
                continue

            added = 0
            for s in submissions:
                pid = s.get("id")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                results.append({
                    "id": pid,
                    "subreddit": s.get("subreddit", sub),
                    "title": s.get("title", ""),
                    "selftext": s.get("selftext", ""),
                    "score": s.get("score", 0),
                    "url": s.get("url", ""),
                    "created_utc": s.get("created_utc"),
                    "num_comments": s.get("num_comments", 0),
                    "search_term": term,
                })
                added += 1

            print(f"+{added} posts (total: {len(results)})", flush=True)
            time.sleep(DELAY)

    return results


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scraping Reddit via Pullpush.io: {SUBREDDITS} x {len(SEARCH_TERMS)} terms...")
    data = scrape()
    OUTPUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Done. {len(data)} unique threads -> {OUTPUT}")


if __name__ == "__main__":
    main()
