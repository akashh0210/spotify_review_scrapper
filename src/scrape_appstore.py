"""Scrape Spotify reviews from Apple App Store via iTunes RSS JSON feed.

No auth required. Loops pages 1-10 across 5 storefronts (us, gb, ca, au, in).
URL pattern: https://itunes.apple.com/{cc}/rss/customerreviews/page={n}/id=324684580/sortby=mostrecent/json
"""

import json
import time
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "appstore.json"
APP_ID = "324684580"
STOREFRONTS = ["us", "gb", "ca", "au", "in"]
PAGES = range(1, 11)
HEADERS = {"User-Agent": "spotify-discovery-research/0.1"}


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch_page(cc: str, page: int) -> list[dict]:
    url = (
        f"https://itunes.apple.com/{cc}/rss/customerreviews"
        f"/page={page}/id={APP_ID}/sortby=mostrecent/json"
    )
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 429:
        raise requests.HTTPError("429 rate-limited")
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    feed = resp.json().get("feed", {})
    entries = feed.get("entry", [])
    if not entries:
        return []
    # First entry on page 1 is app metadata, not a review — filter it out
    return [e for e in entries if "im:rating" in e]


def _parse_entry(entry: dict, cc: str) -> dict:
    return {
        "id": entry.get("id", {}).get("label"),
        "storefront": cc,
        "title": entry.get("title", {}).get("label"),
        "text": entry.get("content", {}).get("label"),
        "rating": entry.get("im:rating", {}).get("label"),
        "vote_count": entry.get("im:voteCount", {}).get("label"),
        "author": entry.get("author", {}).get("name", {}).get("label"),
        "date": entry.get("updated", {}).get("label"),
        "version": entry.get("im:version", {}).get("label"),
    }


def scrape() -> list[dict]:
    seen: set[str] = set()
    results: list[dict] = []

    for cc in STOREFRONTS:
        for page in PAGES:
            try:
                entries = _fetch_page(cc, page)
            except Exception as exc:
                print(f"  [{cc} p{page}] failed: {exc}")
                break

            if not entries:
                print(f"  [{cc} p{page}] empty — stopping this storefront")
                break

            new = 0
            for entry in entries:
                parsed = _parse_entry(entry, cc)
                rid = parsed["id"]
                if rid and rid not in seen:
                    seen.add(rid)
                    results.append(parsed)
                    new += 1

            print(f"  [{cc} p{page}] +{new} reviews (total so far: {len(results)})", flush=True)
            time.sleep(1.0)

    return results


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scraping App Store RSS: app_id={APP_ID}, storefronts={STOREFRONTS}, pages 1-10...")
    data = scrape()
    OUTPUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Done. {len(data)} unique reviews -> {OUTPUT}")


if __name__ == "__main__":
    main()
