"""Scrape community.spotify.com (Khoros platform).

Targets the Discovery & Promo board, Ideas boards, and keyword search results,
filtered to recommendation/discovery topics. Uses requests + BeautifulSoup.
Captures title, body excerpt, kudos count, and thread URL.
Writes data/raw/forum.json.
"""

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "forum.json"
BASE = "https://community.spotify.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY = 2.5
PAGES_PER_BOARD = 5

BOARDS = [
    # Most directly relevant — discovery sharing + algorithm discussions
    "/t5/Discovery-Promo/bd-p/discovery_and_promo",
    # Live and Closed Ideas (IdeaExchange — highest signal)
    "/t5/Live-Ideas/idb-p/liveideas",
    "/t5/Closed-Ideas/idb-p/ideas_no",
    # Content / music questions
    "/t5/Content-Questions/bd-p/content",
]

SEARCH_TERMS = [
    "discover weekly",
    "recommendations algorithm",
    "same songs repeat",
    "daily mix",
    "autoplay radio",
    "new music discovery",
    "filter bubble",
    "repetitive playlist",
]

DISCOVERY_KEYWORDS = {
    "discover", "recommendation", "algorithm", "repeat", "same song",
    "daily mix", "autoplay", "radio", "new music", "stuck", "repetitive",
    "playlist", "suggest", "filter", "genre", "explore", "release radar",
    "discover weekly", "mix",
}


@retry(
    retry=retry_if_exception_type(requests.HTTPError),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _get(url: str, params: dict | None = None) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    if resp.status_code == 429:
        raise requests.HTTPError("429")
    if resp.status_code in (404, 403):
        return BeautifulSoup("", "html.parser")
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _parse_kudos(text: str) -> int:
    m = re.search(r"(\d[\d,]*)", text.replace(",", ""))
    return int(m.group(1)) if m else 0


def _is_relevant(title: str, body: str) -> bool:
    combined = (title + " " + body).lower()
    return any(kw in combined for kw in DISCOVERY_KEYWORDS)


def _extract_threads_from_board(soup: BeautifulSoup) -> list[dict]:
    """Parse thread stubs from a board listing page (article.custom-message-tile)."""
    threads = []
    for article in soup.select("article.custom-message-tile"):
        title_el = article.select_one("h2 a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        href = title_el.get("href", "")
        if not href:
            continue

        # body snippet is the <p> that follows the <h2>
        p_els = article.select("p")
        snippet = p_els[0].get_text(strip=True) if p_els else ""

        kudos_el = article.select_one(".kudos")
        kudos = _parse_kudos(kudos_el.get_text(strip=True)) if kudos_el else 0

        threads.append({
            "title": title,
            "url": urljoin(BASE, href),
            "snippet": snippet,
            "kudos": kudos,
        })
    return threads


def _fetch_thread_body(url: str) -> str:
    """Fetch a thread page and return its full body text."""
    try:
        soup = _get(url)
        # First post on the thread (not replies)
        body_el = soup.select_one(".lia-message-body-content")
        return body_el.get_text(separator=" ", strip=True) if body_el else ""
    except Exception:
        return ""


def _scrape_board(path: str) -> list[dict]:
    results = []
    for page_n in range(1, PAGES_PER_BOARD + 1):
        url = f"{BASE}{path}" if page_n == 1 else f"{BASE}{path}?page={page_n}"
        print(f"  board page: {url}", flush=True)
        try:
            soup = _get(url)
        except Exception as exc:
            print(f"    fetch failed: {exc}")
            break

        threads = _extract_threads_from_board(soup)
        if not threads:
            print(f"    no threads on page {page_n}, stopping")
            break

        for t in threads:
            if not _is_relevant(t["title"], t["snippet"]):
                continue
            time.sleep(DELAY)
            body = _fetch_thread_body(t["url"])
            t["body"] = body
            results.append(t)
            print(f"    + '{t['title'][:60]}' kudos={t['kudos']}", flush=True)

        time.sleep(DELAY)
    return results


def _scrape_search(term: str) -> list[dict]:
    """Search community and extract unique /td-p/ thread links."""
    url = f"{BASE}/t5/forums/searchpage/tab/message"
    try:
        soup = _get(url, params={"q": term, "search_type": "thread"})
    except Exception as exc:
        print(f"    search failed for '{term}': {exc}")
        return []

    # Search result links are /m-p/ replies; extract unique thread links
    thread_links: dict[str, str] = {}  # url -> title
    for a in soup.select("a[href*='/td-p/']"):
        href = urljoin(BASE, a.get("href", ""))
        text = a.get_text(strip=True)
        if href not in thread_links and text:
            thread_links[href] = text

    results = []
    for t_url, title in thread_links.items():
        if not _is_relevant(title, ""):
            continue
        time.sleep(DELAY)
        body = _fetch_thread_body(t_url)
        kudos_soup = _get(t_url)
        kudos_el = kudos_soup.select_one(".kudos")
        kudos = _parse_kudos(kudos_el.get_text(strip=True)) if kudos_el else 0
        results.append({
            "title": title,
            "url": t_url,
            "snippet": "",
            "body": body,
            "kudos": kudos,
        })
        print(f"    + '{title[:60]}' kudos={kudos}", flush=True)
        time.sleep(DELAY)
    return results


def scrape() -> list[dict]:
    seen: set[str] = set()
    all_posts: list[dict] = []

    def _add(posts: list[dict]) -> int:
        added = 0
        for p in posts:
            u = p.get("url", "")
            if u and u not in seen:
                seen.add(u)
                all_posts.append(p)
                added += 1
        return added

    print("== Phase A: targeted boards ==")
    for path in BOARDS:
        n = _add(_scrape_board(path))
        print(f"  {path.split('/')[2]} -> {n} new posts (total: {len(all_posts)})")

    print("== Phase B: keyword search ==")
    for term in SEARCH_TERMS:
        print(f"  searching '{term}'...")
        n = _add(_scrape_search(term))
        print(f"  '{term}' -> {n} new posts (total: {len(all_posts)})")

    return all_posts


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print("Scraping community.spotify.com...")
    data = scrape()
    OUTPUT.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"Done. {len(data)} unique forum posts -> {OUTPUT}")


if __name__ == "__main__":
    main()
