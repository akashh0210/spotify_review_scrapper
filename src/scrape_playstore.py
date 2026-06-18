"""Scrape Spotify reviews from Google Play Store."""

import json
import os
import time
from pathlib import Path

from google_play_scraper import Sort, reviews

OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "playstore.json"
APP_ID = "com.spotify.music"
TARGET = int(os.getenv("PLAYSTORE_REVIEW_COUNT", "3000"))
BATCH = 200


def scrape() -> list[dict]:
    all_reviews = []
    continuation_token = None

    for sort_mode in (Sort.NEWEST, Sort.MOST_RELEVANT):
        fetched = 0
        continuation_token = None
        limit = TARGET // 2

        while fetched < limit:
            count = min(BATCH, limit - fetched)
            result, continuation_token = reviews(
                APP_ID,
                lang="en",
                country="us",
                sort=sort_mode,
                count=count,
                continuation_token=continuation_token,
            )
            if not result:
                break
            all_reviews.extend(result)
            fetched += len(result)
            print(f"  [{sort_mode.name}] fetched {fetched}/{limit}", flush=True)
            if not continuation_token:
                break
            time.sleep(0.5)

    # dedupe by reviewId
    seen = set()
    unique = []
    for r in all_reviews:
        rid = r.get("reviewId")
        if rid and rid not in seen:
            seen.add(rid)
            unique.append(r)

    return unique


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scraping Play Store: {APP_ID} (target ~{TARGET} reviews)...")
    data = scrape()
    OUTPUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Done. {len(data)} unique reviews -> {OUTPUT}")


if __name__ == "__main__":
    main()
