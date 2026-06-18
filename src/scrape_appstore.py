"""Scrape Spotify reviews from Apple App Store (US). Best-effort — failures are non-blocking."""

import json
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "appstore.json"
APP_ID = "324684580"
COUNTRY = "us"


def scrape() -> list[dict]:
    # app-store-scraper has an unstable API across versions; guard defensively.
    try:
        from app_store_scraper import AppStore
    except ImportError:
        print("WARNING: app-store-scraper not importable — skipping App Store.")
        return []

    try:
        app = AppStore(country=COUNTRY, app_name="spotify", app_id=APP_ID)
        app.review(how_many=500)
        return app.reviews if app.reviews else []
    except Exception as exc:
        print(f"WARNING: App Store scrape failed ({exc}) — skipping.")
        return []


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scraping App Store: app_id={APP_ID} country={COUNTRY}...")
    data = scrape()
    if data:
        OUTPUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"Done. {len(data)} reviews -> {OUTPUT}")
    else:
        OUTPUT.write_text("[]", encoding="utf-8")
        print("App Store returned 0 reviews (non-blocking). Empty file written.")


if __name__ == "__main__":
    main()
