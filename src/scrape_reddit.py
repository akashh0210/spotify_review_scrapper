"""Scrape Spotify discovery-related threads from Reddit via PRAW."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
POSTS_PER_TERM = 25  # per subreddit × per term; total ≤ 2 * 10 * 25 = 500 threads
COMMENTS_PER_POST = 20


def _praw_client():
    import praw

    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get(
            "REDDIT_USER_AGENT", "spotify-discovery-engine/1.0"
        ),
    )


def _post_to_dict(post, subreddit: str) -> dict:
    post.comments.replace_more(limit=0)
    top_comments = [
        {"body": c.body, "score": c.score}
        for c in post.comments[:COMMENTS_PER_POST]
        if hasattr(c, "body") and len(c.body) > 20
    ]
    return {
        "id": post.id,
        "subreddit": subreddit,
        "title": post.title,
        "selftext": post.selftext,
        "score": post.score,
        "url": post.url,
        "created_utc": post.created_utc,
        "num_comments": post.num_comments,
        "top_comments": top_comments,
    }


def scrape() -> list[dict]:
    reddit = _praw_client()
    seen_ids: set[str] = set()
    results: list[dict] = []

    for sub_name in SUBREDDITS:
        subreddit = reddit.subreddit(sub_name)
        for term in SEARCH_TERMS:
            print(f"  r/{sub_name} — searching '{term}'...", flush=True)
            try:
                for post in subreddit.search(term, limit=POSTS_PER_TERM, sort="relevance"):
                    if post.id in seen_ids:
                        continue
                    seen_ids.add(post.id)
                    results.append(_post_to_dict(post, sub_name))
            except Exception as exc:
                print(f"    WARNING: search failed for '{term}' in r/{sub_name}: {exc}")

    return results


def main():
    for var in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        if not os.getenv(var):
            raise SystemExit(
                f"ERROR: {var} not set. Copy .env.example → .env and fill in Reddit creds."
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Scraping Reddit: {SUBREDDITS} × {len(SEARCH_TERMS)} terms...")
    data = scrape()
    OUTPUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Done. {len(data)} unique threads -> {OUTPUT}")


if __name__ == "__main__":
    main()
